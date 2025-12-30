from __future__ import annotations
from typing import Dict, List, Sequence, Tuple, Union
import numpy as np

from . import robot_config as rc

Pose6 = Union[List[float], Tuple[float, float, float, float, float, float]]
MeasureDict = Dict[str, Union[int, float, str]]

def ensure_pose6(pose: Sequence[Union[int, float]]) -> List[float]:
    if not isinstance(pose, (list, tuple)) or len(pose) < 6:
        raise ValueError(f"pose6 must be list/tuple len>=6")
    return [float(x) for x in pose[:6]]

def compute_move_xyz_from_measure(measure_res: MeasureDict, current_ry: float) -> Tuple[float, float, float]:
    """
    1단계: 카메라 측정값과 현재 로봇 기울기(RY)를 이용한 기본 이동량 계산
    """
    # 1) cam 측정값 + 오프셋
    cx = float(measure_res["cam_x_mm"]) + float(getattr(rc, "OFF_X_MM", 0.0))
    cy = float(measure_res["cam_y_mm"]) + float(getattr(rc, "OFF_Y_MM", 0.0))
    cz = float(measure_res["cam_z_mm"]) + float(getattr(rc, "OFF_Z_MM", 0.0))

    # 2) 카메라 로컬 부호 매핑 (기존 규칙 유지)
    dx0 = -cx
    dy0 = cy
    dz0 = -cz

    # 3) Pitch (RY) 회전 보정 (카메라 3D 좌표 -> 수평 좌표계로 변환)
    rad_ry = np.deg2rad(current_ry)
    c, s = np.cos(rad_ry), np.sin(rad_ry)
    
    dx1 = dx0
    dy1 = dy0 * c + dz0 * s  # 수평 전진 성분
    dz1 = -dy0 * s + dz0 * c # 수직 하강 성분

    # 4) Yaw (Base Offset) 회전 보정 (-135도 방향 등)
    yaw_deg = float(getattr(rc, "BASE_YAW_OFFSET_DEG", -135.0))
    rad_yaw = np.deg2rad(yaw_deg)
    c_y, s_y = np.cos(rad_yaw), np.sin(rad_yaw)
    
    dx_final = c_y * dx1 - s_y * dy1
    dy_final = s_y * dx1 + c_y * dy1
    dz_final = dz1

    return float(dx_final), float(dy_final), float(dz_final)

def build_target_pose(current_tcp_pose6: Pose6, measure_res: MeasureDict) -> List[float]:
    """
    2단계: 위치 계산 + Pivot 보정 + RZ 각도 보정 통합
    """
    x, y, z, rx, ry, rz = ensure_pose6(current_tcp_pose6)

    # [A] 카메라 데이터를 기반으로 현재 각도에서 가야 할 타겟 XYZ 계산
    dx, dy, dz = compute_move_xyz_from_measure(measure_res, ry)
    target_x = x + dx
    target_y = y + dy
    target_z = z + dz

    # [B] RY 보정: 팔을 0도로 펴면서 발생하는 위치 이탈 보정
    # L(PIVOT_LENGTH): J4 축에서 TCP 끝점까지의 거리
    L = float(getattr(rc, "PIVOT_LENGTH", 165.0)) 
    target_ry = 2.0
    
    rad_curr = np.deg2rad(ry)
    rad_targ = np.deg2rad(target_ry)

    # 팔이 펴질 때 발생하는 로컬 변위 계산 (삼각함수 차이)
    comp_y_local = L * (np.sin(rad_curr) - np.sin(rad_targ))
    comp_z_local = L * (np.cos(rad_targ) - np.cos(rad_curr))

    # 로컬 변위(Y)를 로봇 베이스 방향(Yaw)에 맞춰 X, Y로 분해
    yaw_deg = float(getattr(rc, "BASE_YAW_OFFSET_DEG", -135.0))
    rad_yaw = np.deg2rad(yaw_deg)
    
    comp_dx = -np.sin(rad_yaw) * comp_y_local
    comp_dy = np.cos(rad_yaw) * comp_y_local

    # [C] RZ 각도 보정: 카메라가 측정한 박스 각도 반영
    box_angle = float(measure_res.get("angle_deg", 0.0))
    target_rz = rz - box_angle

    # [최종] 위치에서 보정량을 빼주고 각도를 적용
    final_x = target_x - comp_dx
    final_y = target_y - comp_dy
    final_z = target_z - comp_z_local

    return [final_x, final_y, final_z, rx, target_ry, target_rz]

def build_target_pose_with_debug(current_tcp_pose6: Pose6, measure_res: MeasureDict) -> Dict[str, object]:
    """
    디버그용 정보 포함
    """
    x, y, z, rx, ry, rz = ensure_pose6(current_tcp_pose6)
    target = build_target_pose(current_tcp_pose6, measure_res)
    
    # 실제 반영된 이동량 계산
    move_x = target[0] - x
    move_y = target[1] - y
    move_z = target[2] - z
    
    return {
        "target_pose6": target,
        "move_x_mm": move_x,
        "move_y_mm": move_y,
        "move_z_mm": move_z,
        "current_ry": ry,
        "target_ry": target[4],
        "angle_deg": float(measure_res.get("angle_deg", 0.0)),
    }

def fmt_pose6(pose6: Pose6) -> str:
    x, y, z, rx, ry, rz = ensure_pose6(pose6)
    return f"[x,y,z,rx,ry,rz]=[{x:.3f}, {y:.3f}, {z:.3f}, {rx:.3f}, {ry:.3f}, {rz:.3f}]"