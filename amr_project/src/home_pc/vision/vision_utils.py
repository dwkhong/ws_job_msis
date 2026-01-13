# vision/vision_utils.py
"""
Vision 시스템 유틸리티 함수
- 기하학 계산
- 깊이 통계
- 각도 계산
"""
from typing import Optional, Dict
import numpy as np
import cv2


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