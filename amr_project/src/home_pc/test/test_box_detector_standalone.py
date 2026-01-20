#!/usr/bin/env python3
"""
BoxDetector 독립 테스트 스크립트
카메라와 YOLO 모델만 연결해서 박스 감지 및 스택 카운팅 테스트

사용법:
    python3 test_box_detector_standalone.py
"""

import sys
import time
import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO
from typing import Optional, Dict
import threading

# ============================================================
# 설정 (직접 수정 가능)
# ============================================================

# YOLO 모델 경로
MODEL_PATH = "/home/dw/ws_job_msislab/amr_project/src/job_pc/runs/obb/20260115_obb_test_1/weights/best_obj_20260115.pt"

# 카메라 설정
WIDTH = 640
HEIGHT = 480
FPS = 30

# YOLO 설정
CONF_THRES = 0.85
IOU_THRES = 0.85
IMGSZ = 640

# 스택 카운팅 설정
ENABLE_STACK_COUNTING = True      # True로 변경하면 스택 카운팅 활성화
BASELINE_DEPTH_MM = 570.0         # 빈 테이블 깊이 (실측 필요!)
BOX_HEIGHT_MM = 58.0              # 박스 1개 높이
STACK_COUNT_MAX = 10              # 최대 스택 개수

# Depth 설정
DEPTH_MIN_M = 0.15
DEPTH_MAX_M = 3.00
ROI_MARGIN_PX = 6.0
MIN_ROI_PIXELS = 200        # 120 → 200 (더 많은 픽셀 필요)
MAD_THRES_M = 0.015         # 0.020 → 0.015 (더 엄격하게)

# Preview 설정
SHOW_PREVIEW = True
PREVIEW_WIN_NAME = "Box Detection Test"
FONT_SCALE = 0.6
TEXT_THICKNESS = 2

# 테스트 시간 (초, None이면 무한)
TEST_DURATION_SEC = None  # 예: 30초간 테스트하려면 30


# ============================================================
# 유틸리티 함수들
# ============================================================

def poly_shrink_towards_center(poly, margin_px):
    """폴리곤을 중심으로 축소"""
    cx = np.mean(poly[:, 0])
    cy = np.mean(poly[:, 1])
    out = []
    for px, py in poly:
        dx = px - cx
        dy = py - cy
        dist = np.sqrt(dx*dx + dy*dy)
        if dist < 1e-9:
            out.append([px, py])
            continue
        ratio = max(0.0, (dist - margin_px) / dist)
        nx = cx + dx * ratio
        ny = cy + dy * ratio
        out.append([nx, ny])
    return np.array(out, dtype=np.float32)


def depth_roi_stats(d_u16, depth_scale, poly, depth_min, depth_max):
    """ROI 영역의 Depth 통계 계산"""
    h, w = d_u16.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.int32(poly)], 1)
    
    pixels = d_u16[mask == 1]
    if len(pixels) == 0:
        return 0.0, 0.0, 0
    
    depths_m = pixels.astype(np.float32) * depth_scale
    valid = (depths_m >= depth_min) & (depths_m <= depth_max)
    
    if np.sum(valid) == 0:
        return 0.0, 0.0, 0
    
    valid_depths = depths_m[valid]
    median = np.median(valid_depths)
    mad = np.median(np.abs(valid_depths - median))
    
    return float(median), float(mad), int(np.sum(valid))


def XY_from_pixel_and_Z(px, py, intr, z_m):
    """픽셀 좌표와 깊이로 3D 좌표 계산"""
    x = (px - intr.ppx) / intr.fx * z_m
    y = (py - intr.ppy) / intr.fy * z_m
    return x, y


def obb_angle_deg_upright0_rightplus(poly):
    """OBB 각도 계산"""
    p0, p1, p2, p3 = poly
    
    dx_01 = p1[0] - p0[0]
    dy_01 = p1[1] - p0[1]
    len_01 = np.sqrt(dx_01**2 + dy_01**2)
    
    dx_12 = p2[0] - p1[0]
    dy_12 = p2[1] - p1[1]
    len_12 = np.sqrt(dx_12**2 + dy_12**2)
    
    if len_01 >= len_12:
        dx, dy = dx_01, dy_01
    else:
        dx, dy = dx_12, dy_12
    
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    
    # -90 ~ +90 범위로 정규화
    while angle_deg > 90:
        angle_deg -= 180
    while angle_deg <= -90:
        angle_deg += 180
    
    return float(angle_deg)


# ============================================================
# 메인 감지 루프
# ============================================================

def run_detection():
    """박스 감지 메인 루프"""
    
    print("=" * 60)
    print("BoxDetector 독립 테스트")
    print("=" * 60)
    print(f"\n[설정]")
    print(f"  모델: {MODEL_PATH}")
    print(f"  해상도: {WIDTH}x{HEIGHT}")
    print(f"  스택 카운팅: {ENABLE_STACK_COUNTING}")
    if ENABLE_STACK_COUNTING:
        print(f"  기준 깊이: {BASELINE_DEPTH_MM} mm")
        print(f"  박스 높이: {BOX_HEIGHT_MM} mm")
    print(f"  테스트 시간: {TEST_DURATION_SEC if TEST_DURATION_SEC else '무한'}초")
    print()
    
    # YOLO 모델 로드
    print("YOLO 모델 로딩 중...")
    model = YOLO(MODEL_PATH)
    print("✅ YOLO 모델 로드 완료\n")
    
    # RealSense 설정
    print("RealSense 카메라 초기화 중...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    
    try:
        profile = pipeline.start(config)
    except Exception as e:
        print(f"❌ 카메라 시작 실패: {e}")
        return
    
    # Depth 스케일 가져오기
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    
    # Align 객체 생성
    align = rs.align(rs.stream.color)
    
    # Depth 필터
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    
    print("✅ RealSense 카메라 시작 완료\n")
    
    # Preview 윈도우
    if SHOW_PREVIEW:
        cv2.namedWindow(PREVIEW_WIN_NAME, cv2.WINDOW_NORMAL)
    
    # 통계 변수
    frame_count = 0
    detected_count = 0
    stack_count = 0
    total_box_count = 0
    
    start_time = time.time()
    
    print("=" * 60)
    print("감지 시작 (ESC 키로 종료)")
    print("=" * 60)
    print()
    
    try:
        while True:
            # 시간 체크
            if TEST_DURATION_SEC:
                if time.time() - start_time > TEST_DURATION_SEC:
                    print("\n⏰ 테스트 시간 종료")
                    break
            
            # 프레임 획득
            frames = pipeline.wait_for_frames(1000)
            aligned = align.process(frames)
            
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
            
            # 이미지 변환
            img = np.asanyarray(color_frame.get_data())
            
            # Depth 필터링
            d_frame = spatial.process(depth_frame)
            d_frame = temporal.process(d_frame)
            d_u16 = np.asanyarray(d_frame.get_data())
            
            # 카메라 내부 파라미터
            intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
            
            frame_count += 1
            
            # YOLO 추론
            results = model.predict(
                img,
                imgsz=IMGSZ,
                conf=CONF_THRES,
                iou=IOU_THRES,
                verbose=False,
            )
            r = results[0]
            
            vis = img.copy()
            IMG_CENTER_X = WIDTH / 2.0
            IMG_CENTER_Y = HEIGHT / 2.0
            
            # OBB 결과 처리
            if hasattr(r, "obb") and r.obb is not None:
                candidates = []
                
                for i, conf in enumerate(r.obb.conf):
                    if float(conf) < CONF_THRES:
                        continue
                    
                    poly = r.obb.xyxyxyxy[i].cpu().numpy().reshape(4, 2)
                    
                    # ROI 축소
                    poly_s = poly_shrink_towards_center(poly, ROI_MARGIN_PX)
                    poly_s[:, 0] = np.clip(poly_s[:, 0], 0, WIDTH - 1)
                    poly_s[:, 1] = np.clip(poly_s[:, 1], 0, HEIGHT - 1)
                    
                    # Depth 통계
                    z_m, mad, count = depth_roi_stats(
                        d_u16, depth_scale, poly_s,
                        DEPTH_MIN_M, DEPTH_MAX_M
                    )
                    
                    if z_m > 0 and count > MIN_ROI_PIXELS and (mad <= MAD_THRES_M):
                        cx = float(np.mean(poly[:, 0]))
                        cy = float(np.mean(poly[:, 1]))
                        dist_from_center = float(
                            np.sqrt((cx - IMG_CENTER_X) ** 2 + (cy - IMG_CENTER_Y) ** 2)
                        )
                        angle = obb_angle_deg_upright0_rightplus(poly)
                        
                        # 스택 개수 계산
                        stack_for_this = 0
                        if ENABLE_STACK_COUNTING:
                            depth_mm = float(z_m * 1000.0)
                            depth_diff = BASELINE_DEPTH_MM - depth_mm
                            
                            # 허용 오차 범위 적용 (반 박스 높이)
                            # 예: depth_diff=87 → stack=1 (58*1=58, 범위: 29~87)
                            # 예: depth_diff=145 → stack=2 (58*2=116, 범위: 87~145)
                            stack_for_this = int((depth_diff + BOX_HEIGHT_MM / 2) / BOX_HEIGHT_MM)
                            stack_for_this = max(0, min(stack_for_this, STACK_COUNT_MAX))
                        else:
                            stack_for_this = 1
                        
                        candidates.append({
                            "z_m": float(z_m),
                            "poly": poly,
                            "angle": float(angle),
                            "dist_center": dist_from_center,
                            "cx": cx,
                            "cy": cy,
                            "stack_count": stack_for_this,
                        })
                
                # 통계 업데이트
                if candidates:
                    detected_count = len(candidates)
                    total_box_count = sum(c["stack_count"] for c in candidates)
                    
                    # 중심에 가장 가까운 박스 선택
                    candidates.sort(key=lambda x: x["dist_center"])
                    best = candidates[0]
                    stack_count = best["stack_count"]
                    
                    # 3D 좌표 계산
                    tx, ty = XY_from_pixel_and_Z(best["cx"], best["cy"], intr, best["z_m"])
                    move_x_mm = float(tx * 1000.0)
                    move_y_mm = float(ty * 1000.0)
                    move_z_mm = float(best["z_m"] * 1000.0)
                    
                    # Preview 그리기
                    if SHOW_PREVIEW:
                        # 선택된 박스만 그리기 (녹색)
                        cv2.polylines(vis, [np.int32(best["poly"])], True, (0, 255, 0), 2)
                        cv2.circle(vis, (int(best["cx"]), int(best["cy"])), 5, (0, 255, 0), -1)
                        
                        # 텍스트 표시
                        if ENABLE_STACK_COUNTING:
                            txt = (
                                f"XYZ: {move_x_mm:.0f}, {move_y_mm:.0f}, {move_z_mm:.0f}  "
                                f"ang:{best['angle']:.1f}  OBB: {detected_count}  Total: {total_box_count}"
                            )
                        else:
                            txt = (
                                f"XYZ: {move_x_mm:.0f}, {move_y_mm:.0f}, {move_z_mm:.0f}  "
                                f"ang:{best['angle']:.1f}  Boxes: {detected_count}"
                            )
                        
                        cv2.putText(
                            vis, txt,
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            FONT_SCALE,
                            (0, 255, 0),
                            TEXT_THICKNESS,
                        )
                        
                        # 프레임 정보
                        info_txt = f"Frame: {frame_count}"
                        cv2.putText(
                            vis, info_txt,
                            (10, HEIGHT - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                        )
                    
                    # 콘솔 출력 (1초마다)
                    if frame_count % FPS == 0:
                        print(f"[Frame {frame_count}] "
                              f"OBB: {detected_count}개, "
                              f"Total: {total_box_count}개, "
                              f"선택된 박스: stack={stack_count}, Z={move_z_mm:.1f}mm")
                
                else:
                    detected_count = 0
                    stack_count = 0
                    total_box_count = 0
                    
                    if SHOW_PREVIEW:
                        txt = "No box detected"
                        cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                    FONT_SCALE, (0, 0, 255), TEXT_THICKNESS)
            
            # Preview 표시
            if SHOW_PREVIEW:
                cv2.imshow(PREVIEW_WIN_NAME, vis)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("\n⏹️  사용자가 종료")
                    break
    
    except KeyboardInterrupt:
        print("\n⏹️  Ctrl+C로 종료")
    
    finally:
        # 정리
        pipeline.stop()
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()
        
        print("\n" + "=" * 60)
        print("테스트 종료")
        print("=" * 60)
        print(f"총 프레임: {frame_count}")
        elapsed = time.time() - start_time
        print(f"실행 시간: {elapsed:.1f}초")
        print(f"평균 FPS: {frame_count / elapsed:.1f}")
        print(f"\n마지막 감지:")
        print(f"  OBB 개수: {detected_count}개")
        print(f"  전체 박스: {total_box_count}개")
        if ENABLE_STACK_COUNTING:
            print(f"  선택된 박스 스택: {stack_count}개")


# ============================================================
# 메인 실행
# ============================================================

if __name__ == "__main__":
    run_detection()