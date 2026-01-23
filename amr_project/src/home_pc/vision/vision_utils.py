# vision/vision_utils.py
"""
Vision 시스템 유틸리티 함수
- 기하학 계산
- 깊이 통계
- 각도 계산
- ArUco 마커 감지
"""
from typing import Optional, Dict
import numpy as np
import cv2
from config import vision_config as cfg


# ============================================================
# ArUco 마커 관련 함수
# ============================================================

def init_aruco_detector():
    """
    ArUco 감지기 초기화
    
    Returns:
        tuple: (aruco_dict, aruco_params) or (None, None)
    """
    try:
        # ArUco 딕셔너리 설정
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        
        # DetectorParameters 생성 (OpenCV 버전 호환)
        try:
            aruco_params = cv2.aruco.DetectorParameters()
            print("[ArUco] DetectorParameters() 사용")
        except:
            aruco_params = cv2.aruco.DetectorParameters_create()
            print("[ArUco] DetectorParameters_create() 사용")
        
        print(f"[ArUco] Initialized with DICT_4X4_50")
        return aruco_dict, aruco_params
        
    except Exception as e:
        print(f"[ArUco] Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def detect_aruco_markers(color_image, depth_image, aruco_dict, aruco_params):
    """
    ArUco 마커 감지 및 Depth 측정
    
    Args:
        color_image: BGR 이미지
        depth_image: Depth 이미지 (미터 단위)
        aruco_dict: ArUco 딕셔너리
        aruco_params: ArUco 파라미터
    
    Returns:
        dict: {marker_id: {'center': (x,y), 'depth_m': z, 'corners': [...]}}
    """
    if aruco_dict is None or aruco_params is None:
        return {}
    
    markers = {}
    
    try:
        # 그레이스케일 변환
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        # 마커 감지
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, 
            aruco_dict, 
            parameters=aruco_params
        )
        
        # 변화가 있을 때만 출력
        if not hasattr(detect_aruco_markers, '_last_ids'):
            detect_aruco_markers._last_ids = None
        
        current_ids = ids.flatten().tolist() if ids is not None else []
        if current_ids != detect_aruco_markers._last_ids:
            if len(current_ids) > 0:
                print(f"[ArUco] Detected {len(current_ids)} markers: {current_ids}")
            else:
                if detect_aruco_markers._last_ids is not None and len(detect_aruco_markers._last_ids) > 0:
                    print(f"[ArUco] No markers detected")
            detect_aruco_markers._last_ids = current_ids
        
        if ids is None or len(ids) == 0:
            return {}
        
        # 각 마커 처리 (config에 설정된 마커만)
        marker_baselines = getattr(cfg, 'ARUCO_MARKER_BASELINES', {1: 560.0, 2: 750.0, 3: 1000.0})
        valid_marker_ids = list(marker_baselines.keys())
        
        for i, marker_id in enumerate(ids.flatten()):
            # config에 설정된 마커만 처리
            if marker_id not in valid_marker_ids:
                continue
            
            marker_corners = corners[i][0]  # shape: (4, 2)
            
            # 마커 중심 계산
            center_x = int(np.mean(marker_corners[:, 0]))
            center_y = int(np.mean(marker_corners[:, 1]))
            
            # ROI 영역에서 Depth 측정
            roi_size = getattr(cfg, 'ARUCO_ROI_SIZE', 20)
            x1 = max(0, center_x - roi_size)
            x2 = min(depth_image.shape[1], center_x + roi_size)
            y1 = max(0, center_y - roi_size)
            y2 = min(depth_image.shape[0], center_y + roi_size)
            
            roi_depth = depth_image[y1:y2, x1:x2]
            
            # 유효한 Depth 값 필터링
            depth_min = getattr(cfg, 'DEPTH_MIN_M', 0.15)
            depth_max = getattr(cfg, 'DEPTH_MAX_M', 3.0)
            valid_depths = roi_depth[
                (roi_depth > depth_min) & 
                (roi_depth < depth_max)
            ]
            
            if len(valid_depths) < 10:
                continue
            
            # 중앙값 사용 (노이즈에 강함)
            marker_depth_m = float(np.median(valid_depths))
            
            markers[int(marker_id)] = {
                'center': (center_x, center_y),
                'depth_m': marker_depth_m,
                'corners': marker_corners.tolist()
            }
        
        return markers
        
    except Exception as e:
        print(f"[ArUco] Detection error: {e}")
        return {}


def get_baseline_from_markers(markers, current_table=None):
    """
    ArUco 마커로부터 BASELINE 결정
    
    마커 ID에 따라 vision_config.ARUCO_MARKER_BASELINES에서 값 가져옴
    
    Args:
        markers: detect_aruco_markers 결과
        current_table: 현재 테이블 번호 (사용 안 함)
    
    Returns:
        float or None: BASELINE (mm), 마커 없으면 None
    """
    if not markers:
        return None
    
    # Config에서 마커별 BASELINE 가져오기
    marker_baselines = getattr(cfg, 'ARUCO_MARKER_BASELINES', {
        1: 560.0,
        2: 750.0,
        3: 1000.0
    })
    
    # 마커 ID 확인
    marker_ids = list(markers.keys())
    
    # BASELINE이 설정된 마커만 필터링
    valid_markers = [mid for mid in marker_ids if mid in marker_baselines]
    
    if not valid_markers:
        print(f"[ArUco] 마커 {marker_ids} 감지되었으나 설정된 마커 없음 (무시)")
        return None
    
    # 가장 큰 ID (우선순위: 숫자가 클수록 높음)
    selected_id = max(valid_markers)
    baseline_mm = marker_baselines[selected_id]
    
    print(f"[ArUco] 마커 {selected_id}번 감지 → BASELINE={baseline_mm}mm")
    return baseline_mm


# 기하학 함수
# ============================================================

def poly_shrink_towards_center(poly4x2: np.ndarray, margin_px: float) -> np.ndarray:
    """
    폴리곤을 중심으로 축소
    
    Args:
        poly4x2: 4x2 폴리곤 좌표
        margin_px: 축소 마진 (픽셀)
    
    Returns:
        축소된 폴리곤
    """
    p = poly4x2.astype(np.float32)
    c = p.mean(axis=0, keepdims=True)
    v = p - c
    norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-6
    return p - (v / norm) * margin_px


def depth_roi_stats(depth_u16: np.ndarray, depth_scale: float, poly4x2: np.ndarray, 
                    depth_min: float, depth_max: float) -> tuple:
    """
    ROI 내 깊이 통계 계산
    
    Args:
        depth_u16: 깊이 이미지 (uint16)
        depth_scale: 깊이 스케일
        poly4x2: ROI 폴리곤
        depth_min: 최소 깊이 (m)
        depth_max: 최대 깊이 (m)
    
    Returns:
        (median, mad, count) 튜플
    """
    h, w = depth_u16.shape[:2]
    poly = np.round(poly4x2).astype(np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly.reshape(-1, 1, 2)], 255)

    d = depth_u16[mask == 255].astype(np.float32) * depth_scale
    d = d[(d > 0) & (d >= depth_min) & (d <= depth_max)]

    if d.size == 0:
        return 0.0, 0.0, 0

    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    return med, mad, int(d.size)


def XY_from_pixel_and_Z(cx: float, cy: float, intr, Z_m: float) -> tuple:
    """
    픽셀 좌표와 깊이로부터 3D 좌표 계산
    
    Args:
        cx, cy: 픽셀 좌표
        intr: 카메라 intrinsics
        Z_m: 깊이 (m)
    
    Returns:
        (X, Y) 좌표 (m)
    """
    X = (cx - intr.ppx) / intr.fx * Z_m
    Y = (cy - intr.ppy) / intr.fy * Z_m
    return float(X), float(Y)


def obb_angle_deg_upright0_rightplus(poly4x2: np.ndarray) -> float:
    """
    OBB 각도 계산 (PCA 기반)
    
    Args:
        poly4x2: 4x2 폴리곤 좌표
    
    Returns:
        각도 (도)
    """
    p = poly4x2.astype(np.float32)
    c = p.mean(axis=0, keepdims=True)
    q = p - c
    cov = np.cov(q.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    v = eigvecs[:, np.argmax(eigvals)].astype(np.float32)
    vx, vy = float(v[0]), float(v[1])
    if vy < 0:
        vx, vy = -vx, -vy
    return -float(np.degrees(np.arctan2(vx, vy)))


def is_jump(prev: Optional[Dict[str, float]], cur: Dict[str, float],
            jump_xy_mm: float, jump_z_mm: float, jump_ang_deg: float) -> bool:
    """
    측정값이 급격히 변했는지 확인 (점프 감지)
    
    Args:
        prev: 이전 측정값
        cur: 현재 측정값
        jump_xy_mm: XY 점프 임계값
        jump_z_mm: Z 점프 임계값
        jump_ang_deg: 각도 점프 임계값
    
    Returns:
        점프가 감지되면 True
    """
    if prev is None:
        return False
    
    if abs(cur["move_x_mm"] - prev["move_x_mm"]) > jump_xy_mm:
        return True
    if abs(cur["move_y_mm"] - prev["move_y_mm"]) > jump_xy_mm:
        return True
    if abs(cur["move_z_mm"] - prev["move_z_mm"]) > jump_z_mm:
        return True
    if abs(cur.get("angle_deg", 0.0) - prev.get("angle_deg", 0.0)) > jump_ang_deg:
        return True
    
    return False