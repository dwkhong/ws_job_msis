# vision/box_detector.py
"""
박스 탐지 시스템 (YOLO OBB + RealSense D405)
실시간 박스 탐지 및 3D 위치 측정
"""
from __future__ import annotations
from typing import Optional, Dict
import os
import time
import threading
import traceback

import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO

from config import vision_config as cfg
from .vision_utils import (
    poly_shrink_towards_center,
    depth_roi_stats,
    XY_from_pixel_and_Z,
    obb_angle_deg_upright0_rightplus,
    is_jump
)


class BoxDetector:
    """
    박스 탐지 및 3D 위치 측정 클래스
    - YOLO OBB 모델로 박스 탐지
    - RealSense D405로 깊이 측정
    - 실시간 스레드 기반 처리
    """
    
    def __init__(self):
        """초기화"""
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._model: Optional[YOLO] = None
        
        # 측정 데이터
        self._latest_sample: Optional[Dict[str, float]] = None
        self._collect_samples: list[Dict[str, float]] = []
        self._last_measure_avg: Optional[Dict[str, float]] = None  # 마지막 평균 측정값 캐시
        
        # 감지된 박스 개수
        self._detected_box_count: int = 0
        
        # 스택 카운팅 (Depth 기반)
        self._stack_count: int = 0  # 선택된 박스의 스택 높이
        self._total_box_count: int = 0  # 전체 박스 개수 (모든 OBB 스택 합산)
        
        # 스레드 동기화
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._measuring_evt = threading.Event()
        self._reset_evt = threading.Event()
        
        # RealSense 관련
        self._pipeline: Optional[rs.pipeline] = None
        self._align: Optional[rs.align] = None
        self._depth_scale: float = 0.001
        self._stream_w = int(cfg.WIDTH)
        self._stream_h = int(cfg.HEIGHT)
        
        # RealSense 호출 직렬화
        self._rs_lock = threading.RLock()
        
        # 타임아웃 설정
        self._wait_ms = 1000
        self._log_every = 5
    
    def is_running(self) -> bool:
        """
        실행 중인지 확인
        
        Returns:
            bool: 실행 중이면 True
        """
        th = self._thread
        return bool(self._running and th is not None and th.is_alive())
    
    def start(self) -> bool:
        """
        비전 시스템 시작
        - YOLO 모델 로드
        - RealSense 파이프라인 시작
        - 탐지 스레드 시작
        
        Returns:
            bool: 시작 성공 시 True
        """
        if self._running:
            print("[Vision] Already running")
            return True
        
        # DISPLAY 확인 (preview용)
        if bool(getattr(cfg, "SHOW_PREVIEW", False)) and os.environ.get("DISPLAY", "") == "":
            print("[Vision] DISPLAY가 없어 preview를 자동으로 끕니다.")
        
        # YOLO 모델 로드
        print(f"[Vision] YOLO 모델 로딩... ({cfg.MODEL_PATH})")
        try:
            self._model = YOLO(cfg.MODEL_PATH, task="obb")
        except Exception as e:
            print(f"[Vision] ERROR: 모델 로드 실패: {e}")
            return False
        
        # 이벤트 초기화
        self._stop_evt.clear()
        self._measuring_evt.clear()
        self._reset_evt.clear()
        
        # 스레드 시작
        self._running = True
        self._thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._thread.start()
        
        print("[Vision] 시스템 시작 ✅")
        return True
    
    def stop(self) -> bool:
        """
        비전 시스템 종료
        
        Returns:
            bool: 종료 성공 시 True
        """
        if not self._running:
            print("[Vision] Not running")
            return False
        
        print("[Vision] 시스템 종료 중...")
        self._running = False
        self._stop_evt.set()
        
        # 스레드 종료 대기
        th = self._thread
        if th is not None:
            th.join(timeout=3.0)
        self._thread = None
        
        # RealSense 파이프라인 종료
        self._safe_stop_pipeline()
        
        # OpenCV 윈도우 정리
        if bool(getattr(cfg, "SHOW_PREVIEW", False)):
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        
        print("[Vision] 시스템 종료 ✅")
        return True
    
    def toggle(self) -> bool:
        """
        시스템 토글 (시작 <-> 종료)
        
        Returns:
            bool: 토글 후 실행 상태
        """
        if self.is_running():
            self.stop()
            return False
        else:
            return self.start()
    
    def measure_avg(self, n: Optional[int] = None, 
                    timeout: Optional[float] = None) -> Optional[Dict[str, float]]:
        """
        평균 측정값 계산
        
        Args:
            n: 샘플 개수 (기본값: cfg.AVG_N)
            timeout: 타임아웃 (초, 기본값: cfg.TIMEOUT_SEC)
        
        Returns:
            측정 결과 딕셔너리 또는 None
            {'move_x_mm', 'move_y_mm', 'move_z_mm', 'angle_deg'}
        """
        if not self.is_running():
            print("[Vision] Not running. Start first.")
            return None
        
        n = int(cfg.AVG_N if n is None else n)
        timeout = float(cfg.TIMEOUT_SEC if timeout is None else timeout)
        
        # 샘플 수집 초기화
        with self._lock:
            self._collect_samples = []
        
        self._measuring_evt.set()
        self._reset_evt.set()
        
        print(f"[Vision] 측정 시작 (목표: {n}개, timeout: {timeout}s)")
        start_t = time.time()
        
        # 샘플 수집 대기
        while time.time() - start_t < timeout:
            with self._lock:
                cnt = len(self._collect_samples)
            if cnt >= n:
                break
            time.sleep(0.02)
        
        self._measuring_evt.clear()
        
        # 수집된 샘플 가져오기
        with self._lock:
            samples = list(self._collect_samples)
        
        if len(samples) == 0:
            print("[Vision] 측정 실패: 데이터 없음")
            return None
        
        if len(samples) < n:
            print(f"[Vision] 경고: 목표 {n}개 중 {len(samples)}개만 수집됨")
        
        # 평균 계산
        arr = np.array(
            [[s["move_x_mm"], s["move_y_mm"], s["move_z_mm"], s["angle_deg"]] 
             for s in samples],
            dtype=np.float32,
        )
        m = np.mean(arr, axis=0)
        
        result = {
            "move_x_mm": float(m[0]),
            "move_y_mm": float(m[1]),
            "move_z_mm": float(m[2]),
            "angle_deg": float(m[3]),
        }
        
        print(f"[Vision] 측정 완료: "
              f"XYZ=({result['move_x_mm']:.1f}, {result['move_y_mm']:.1f}, "
              f"{result['move_z_mm']:.1f}mm), angle={result['angle_deg']:.1f}°")
        
        return result
    
    def cmd_measure_avg(self, n: Optional[int] = None, 
                        timeout: Optional[float] = None) -> Optional[Dict[str, float]]:
        """
        평균 측정 (3번 메뉴용) - 결과를 캐시에 저장
        
        Args:
            n: 샘플 개수
            timeout: 타임아웃 (초)
        
        Returns:
            측정 결과 또는 None
        """
        if not self.is_running():
            print("[Vision] Not running. (1번으로 Vision On)")
            return None
        
        result = self.measure_avg(n=n, timeout=timeout)
        
        if result is None:
            print("[Vision] measure_avg failed (no samples)")
            self._last_measure_avg = None
            return None
        
        # 캐시에 저장
        self._last_measure_avg = result
        
        print(f"[Vision] avg XYZ(mm)=({result['move_x_mm']:.1f}, "
              f"{result['move_y_mm']:.1f}, {result['move_z_mm']:.1f}) "
              f"angle(deg)={result['angle_deg']:.2f}")
        
        return result
    
    def get_last_measure_avg(self) -> Optional[Dict[str, float]]:
        """
        마지막 평균 측정값 반환
        
        Returns:
            마지막 측정값 또는 None
        """
        return self._last_measure_avg
    
    def get_detected_box_count(self) -> int:
        """
        현재 감지된 박스 개수 반환
        
        Returns:
            감지된 박스 개수 (0 이상)
        """
        with self._lock:
            return self._detected_box_count
    
    def get_stack_count(self) -> int:
        """
        Depth 기반 스택 개수 반환 (선택된 박스 위치)
        
        Returns:
            스택된 박스 개수 (0 이상)
        """
        with self._lock:
            return self._stack_count
    
    def get_total_box_count(self) -> int:
        """
        전체 박스 개수 반환 (모든 OBB의 스택 합산)
        
        Returns:
            전체 박스 개수 (0 이상)
        """
        with self._lock:
            return self._total_box_count
    
    # -------------------------
    # RealSense 파이프라인 관리
    # -------------------------
    def _safe_stop_pipeline(self):
        """RealSense 파이프라인 안전하게 종료"""
        with self._rs_lock:
            try:
                if self._pipeline is not None:
                    try:
                        self._pipeline.stop()
                    except Exception:
                        pass
            finally:
                self._pipeline = None
                self._align = None
    
    def _start_pipeline(self) -> bool:
        """
        RealSense 파이프라인 시작
        
        Returns:
            bool: 시작 성공 시 True
        """
        with self._rs_lock:
            self._safe_stop_pipeline()
            time.sleep(0.2)
            
            pipeline = rs.pipeline()
            config = rs.config()
            
            STREAM_W = int(cfg.WIDTH)
            STREAM_H = int(cfg.HEIGHT)
            FPS = int(cfg.FPS)
            
            config.enable_stream(rs.stream.color, STREAM_W, STREAM_H, rs.format.bgr8, FPS)
            config.enable_stream(rs.stream.depth, STREAM_W, STREAM_H, rs.format.z16, FPS)
            
            try:
                profile = pipeline.start(config)
            except Exception as e:
                print(f"[Vision] ERROR: RealSense start failed: {e}")
                try:
                    pipeline.stop()
                except Exception:
                    pass
                return False
            
            self._pipeline = pipeline
            self._align = rs.align(rs.stream.color)
            self._depth_scale = float(
                profile.get_device().first_depth_sensor().get_depth_scale()
            )
            self._stream_w = STREAM_W
            self._stream_h = STREAM_H
            
            print(f"[Vision] RealSense 시작 ({STREAM_W}x{STREAM_H} @ {FPS}fps)")
            return True
    
    # -------------------------
    # 메인 탐지 루프
    # -------------------------
    def _detection_loop(self):
        """
        실시간 박스 탐지 루프 (스레드에서 실행)
        """
        last_log_t = 0.0
        fail_cnt = 0
        
        try:
            # RealSense 파이프라인 시작
            if not self._start_pipeline():
                self._running = False
                return
            
            # Depth 필터 설정
            spatial = rs.spatial_filter()
            temporal = rs.temporal_filter()
            
            # Preview 윈도우 생성
            if bool(getattr(cfg, "SHOW_PREVIEW", False)):
                try:
                    cv2.namedWindow(cfg.PREVIEW_WIN_NAME, cv2.WINDOW_NORMAL)
                except Exception:
                    print("[Vision] Preview window 생성 실패")
            
            prev_valid: Optional[Dict[str, float]] = None
            
            # 설정값 로드
            roi_margin_px = float(getattr(cfg, "ROI_MARGIN_PX", 5.0))
            min_roi_pixels = int(getattr(cfg, "MIN_ROI_PIXELS", 50))
            mad_thres_m = float(getattr(cfg, "MAD_THRES_M", 0.020))
            
            # 메인 루프
            while self._running and (not self._stop_evt.is_set()):
                # 리셋 이벤트 처리
                if self._reset_evt.is_set():
                    prev_valid = None
                    self._reset_evt.clear()
                
                try:
                    # 프레임 획득
                    with self._rs_lock:
                        pipeline = self._pipeline
                        align = self._align
                        if pipeline is None or align is None:
                            time.sleep(0.05)
                            continue
                        
                        frames = pipeline.wait_for_frames(self._wait_ms)
                        aligned = align.process(frames)
                        
                        color_frame = aligned.get_color_frame()
                        depth_frame = aligned.get_depth_frame()
                        if not color_frame or not depth_frame:
                            continue
                        
                        img = np.asanyarray(color_frame.get_data())
                        
                        # Depth 필터링
                        d_frame = spatial.process(depth_frame)
                        d_frame = temporal.process(d_frame)
                        d_u16 = np.asanyarray(d_frame.get_data())
                        
                        intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
                        STREAM_W = int(self._stream_w)
                        STREAM_H = int(self._stream_h)
                    
                    fail_cnt = 0
                    
                    IMG_CENTER_X = STREAM_W / 2.0
                    IMG_CENTER_Y = STREAM_H / 2.0
                    
                    vis = img.copy()
                    
                    # YOLO 추론
                    results = self._model.predict(
                        img,
                        imgsz=int(cfg.IMGSZ),
                        conf=float(cfg.CONF_THRES),
                        iou=float(cfg.IOU_THRES),
                        verbose=False,
                    )
                    r = results[0]
                    
                    best_sample: Optional[Dict[str, float]] = None
                    
                    # OBB 결과 처리
                    if hasattr(r, "obb") and r.obb is not None:
                        candidates = []
                        
                        for i, conf in enumerate(r.obb.conf):
                            if float(conf) < float(cfg.CONF_THRES):
                                continue
                            
                            poly = r.obb.xyxyxyxy[i].cpu().numpy().reshape(4, 2)
                            
                            # ROI 축소 및 클리핑
                            poly_s = poly_shrink_towards_center(poly, roi_margin_px)
                            poly_s[:, 0] = np.clip(poly_s[:, 0], 0, STREAM_W - 1)
                            poly_s[:, 1] = np.clip(poly_s[:, 1], 0, STREAM_H - 1)
                            
                            # Depth 통계 계산
                            z_m, mad, count = depth_roi_stats(
                                d_u16, float(self._depth_scale), poly_s,
                                cfg.DEPTH_MIN_M, cfg.DEPTH_MAX_M
                            )
                            
                            # 유효성 검사
                            if z_m > 0 and count > min_roi_pixels and (mad <= mad_thres_m):
                                cx = float(np.mean(poly[:, 0]))
                                cy = float(np.mean(poly[:, 1]))
                                dist_from_center = float(
                                    np.sqrt((cx - IMG_CENTER_X) ** 2 + (cy - IMG_CENTER_Y) ** 2)
                                )
                                angle = obb_angle_deg_upright0_rightplus(poly)
                                
                                # 각 OBB의 스택 개수 계산
                                stack_for_this = 0
                                if bool(getattr(cfg, "ENABLE_STACK_COUNTING", False)):
                                    box_h = float(getattr(cfg, "BOX_HEIGHT_MM", 58.0))
                                    depth_mm = float(z_m * 1000.0)
                                    
                                    # 다중 테이블 높이 지원
                                    if bool(getattr(cfg, "USE_MULTI_BASELINE", False)):
                                        baseline_low = float(getattr(cfg, "BASELINE_DEPTH_LOW_MM", 560.0))
                                        baseline_high = float(getattr(cfg, "BASELINE_DEPTH_HIGH_MM", 750.0))
                                        
                                        # OBB 주변 바닥 Depth 측정
                                        surrounding_depth_mm = None
                                        try:
                                            # poly를 바깥쪽으로 확장 (30px)
                                            poly_np = np.array(poly, dtype=np.float32)
                                            center = poly_np.mean(axis=0)
                                            expanded_poly = center + (poly_np - center) * 1.5
                                            
                                            # 확장된 영역과 원본 영역 마스크 생성
                                            mask_outer = np.zeros((STREAM_H, STREAM_W), dtype=np.uint8)
                                            mask_inner = np.zeros((STREAM_H, STREAM_W), dtype=np.uint8)
                                            cv2.fillPoly(mask_outer, [np.int32(expanded_poly)], 1)
                                            cv2.fillPoly(mask_inner, [np.int32(poly)], 1)
                                            
                                            # 링 영역 (outer - inner)
                                            mask_ring = mask_outer - mask_inner
                                            
                                            # 링 영역의 Depth 값 추출
                                            ring_depths = d_u16[mask_ring == 1]
                                            
                                            if len(ring_depths) > 10:  # 최소 10픽셀 이상
                                                # 유효한 Depth만 (0이 아닌 값)
                                                valid_depths = ring_depths[ring_depths > 0]
                                                if len(valid_depths) > 5:
                                                    # 중앙값 사용 (노이즈에 강함)
                                                    median_depth_unit = float(np.median(valid_depths))
                                                    # RealSense depth scale (보통 0.001)
                                                    depth_scale = d_frame.get_units()
                                                    surrounding_depth_mm = median_depth_unit * depth_scale * 1000.0
                                        except Exception as e:
                                            # 측정 실패 시 None
                                            surrounding_depth_mm = None
                                        
                                        # BASELINE 선택
                                        if surrounding_depth_mm is not None:
                                            # 주변 바닥 측정 성공 - 가까운 BASELINE 선택
                                            diff_low = abs(surrounding_depth_mm - baseline_low)
                                            diff_high = abs(surrounding_depth_mm - baseline_high)
                                            
                                            if diff_low < diff_high:
                                                baseline = baseline_low
                                            else:
                                                baseline = baseline_high
                                        else:
                                            # 측정 실패 - fallback: 박스 Z값으로 판단
                                            threshold = float(getattr(cfg, "BASELINE_THRESHOLD_MM", 650.0))
                                            if depth_mm > threshold:
                                                baseline = baseline_high
                                            else:
                                                baseline = baseline_low
                                    else:
                                        # 단일 테이블 모드
                                        baseline = float(getattr(cfg, "BASELINE_DEPTH_MM", 560.0))
                                    
                                    depth_diff = baseline - depth_mm  # 박스 쌓이면 Z 감소
                                    
                                    # 허용 오차 범위 적용 (반 박스 높이)
                                    stack_for_this = int((depth_diff + box_h / 2) / box_h)
                                    stack_for_this = max(0, min(stack_for_this, int(getattr(cfg, "STACK_COUNT_MAX", 10))))
                                else:
                                    stack_for_this = 1  # 스택 카운팅 비활성화 시 각 OBB = 1개
                                
                                candidates.append({
                                    "z_m": float(z_m),
                                    "poly": poly,
                                    "angle": float(angle),
                                    "dist_center": dist_from_center,
                                    "cx": cx,
                                    "cy": cy,
                                    "stack_count": stack_for_this,  # 이 OBB의 스택 개수

                                })
                        
                        # 가장 중심에 가까운 박스 선택
                        if candidates:
                            # 전체 감지된 박스 개수 저장
                            with self._lock:
                                self._detected_box_count = len(candidates)
                            
                            # 전체 박스 개수 계산 (모든 OBB의 스택 합산)
                            total_boxes = sum(c["stack_count"] for c in candidates)
                            with self._lock:
                                self._total_box_count = total_boxes
                            
                            # 박스 선택 우선순위 계산
                            z_threshold_mm = float(getattr(cfg, "BOX_SELECTION_Z_THRESHOLD_MM", 30.0))
                            min_z = min(c["z_m"] for c in candidates)
                            
                            for c in candidates:
                                z_diff_mm = (c["z_m"] - min_z) * 1000.0
                                
                                if z_diff_mm > z_threshold_mm:
                                    # Z 차이가 크면 높은 박스 우선 (큰 페널티)
                                    c["priority"] = z_diff_mm * 10.0 + c["dist_center"]
                                else:
                                    # Z 차이가 작으면 XY 중심 거리만 고려
                                    c["priority"] = c["dist_center"]
                            
                            # 우선순위로 정렬 (낮은 값이 우선)
                            candidates.sort(key=lambda x: x["priority"])
                            best = candidates[0]
                            
                            # Preview: 모든 candidates를 빨간색으로 그리기
                            if bool(getattr(cfg, "SHOW_PREVIEW", False)):
                                for c in candidates:
                                    cv2.polylines(vis, [np.int32(c["poly"])], True, (0, 0, 255), 1)
                            
                            # 3D 좌표 계산
                            tx, ty = XY_from_pixel_and_Z(best["cx"], best["cy"], intr, best["z_m"])
                            raw = {
                                "move_x_mm": float(tx * 1000.0),
                                "move_y_mm": float(ty * 1000.0),
                                "move_z_mm": float(best["z_m"] * 1000.0),
                                "angle_deg": float(best["angle"]),
                            }
                            
                            # Depth 기반 스택 카운팅 (선택된 박스)
                            if bool(getattr(cfg, "ENABLE_STACK_COUNTING", False)):
                                # 선택된 박스의 스택 개수 저장
                                with self._lock:
                                    self._stack_count = best["stack_count"]
                            else:
                                with self._lock:
                                    self._stack_count = 0
                            
                            # Jump 필터링
                            if not is_jump(
                                prev_valid, raw,
                                cfg.JUMP_XY_MM, cfg.JUMP_Z_MM, cfg.JUMP_ANG_DEG
                            ):
                                prev_valid = raw
                                best_sample = raw
                                
                                # Preview 그리기 (Jump 통과한 박스만)
                                if bool(getattr(cfg, "SHOW_PREVIEW", False)):
                                    # 박스와 중심점 그리기
                                    cv2.polylines(vis, [np.int32(best["poly"])], True, (0, 255, 0), 2)
                                    cv2.circle(vis, (int(best["cx"]), int(best["cy"])), 5, (0, 255, 0), -1)
                                    
                                    # XYZ 좌표는 박스 옆에 표시
                                    xyz_txt = (
                                        f"XYZ: {raw['move_x_mm']:.0f}, {raw['move_y_mm']:.0f}, "
                                        f"{raw['move_z_mm']:.0f}  ang:{raw['angle_deg']:.1f}"
                                    )
                                    cv2.putText(
                                        vis, xyz_txt,
                                        (max(0, int(best["cx"]) - 20), max(20, int(best["cy"]) - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        float(getattr(cfg, "OVERLAY_FONT_SCALE", 0.6)),
                                        (0, 255, 0),
                                        int(getattr(cfg, "OVERLAY_THICKNESS", 2)),
                                    )
                            
                            # OBB/Total은 Jump 여부와 관계없이 항상 표시
                            if bool(getattr(cfg, "SHOW_PREVIEW", False)):
                                if bool(getattr(cfg, "ENABLE_STACK_COUNTING", False)):
                                    count_txt = f"OBB: {len(candidates)}  Total: {total_boxes}"
                                else:
                                    count_txt = f"Boxes: {len(candidates)}"
                                
                                # 텍스트 크기 계산
                                text_size = cv2.getTextSize(
                                    count_txt, 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.5, 
                                    1
                                )[0]
                                
                                # 오른쪽 위 좌표 (여백 10px)
                                text_x = STREAM_W - text_size[0] - 10
                                text_y = 20
                                
                                cv2.putText(
                                    vis, count_txt,
                                    (text_x, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 255, 255),  # 노란색
                                    1,
                                )
                        else:
                            # 박스가 없으면 0으로 설정
                            with self._lock:
                                self._detected_box_count = 0
                                self._stack_count = 0
                                self._total_box_count = 0
                    
                    # 샘플 저장
                    if best_sample is not None:
                        with self._lock:
                            self._latest_sample = best_sample
                            if self._measuring_evt.is_set():
                                self._collect_samples.append(best_sample)
                    
                    # Preview 표시
                    if bool(getattr(cfg, "SHOW_PREVIEW", False)):
                        cv2.imshow(cfg.PREVIEW_WIN_NAME, vis)
                        if cv2.waitKey(1) & 0xFF == 27:  # ESC
                            break
                
                except RuntimeError as e:
                    # 프레임 타임아웃 처리
                    se = str(e).lower()
                    if ("frame didn't arrive" in se) or ("frame didn" in se and "arrive" in se):
                        fail_cnt += 1
                        if fail_cnt == 1 or (fail_cnt % self._log_every == 0):
                            print(f"[Vision] Frame timeout x{fail_cnt}")
                        time.sleep(0.02)
                        continue
                    
                    # 기타 RuntimeError
                    now = time.time()
                    if now - last_log_t > 2.0:
                        print("[Vision] RuntimeError:")
                        print(traceback.format_exc())
                        last_log_t = now
                    time.sleep(0.01)
                
                except Exception:
                    # 기타 예외
                    now = time.time()
                    if now - last_log_t > 2.0:
                        print("[Vision] Exception:")
                        print(traceback.format_exc())
                        last_log_t = now
                    time.sleep(0.01)
        
        finally:
            # 정리
            self._safe_stop_pipeline()
            if bool(getattr(cfg, "SHOW_PREVIEW", False)):
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass