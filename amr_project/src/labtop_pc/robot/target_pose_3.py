from __future__ import annotations
from typing import Dict, List, Sequence, Tuple, Union, Any
import numpy as np
from . import robot_config as rc

Pose6 = Union[List[float], Tuple[float, float, float, float, float, float]]
MeasureDict = Dict[str, Union[int, float, str]]

def ensure_pose6(pose: Sequence[Union[int, float]]) -> List[float]:
    if not isinstance(pose, (list, tuple)) or len(pose) < 6:
        raise ValueError(f"pose6 must be list/tuple len>=6")
    return [float(x) for x in pose[:6]]

def get_rotation_matrix_z(deg: float) -> np.ndarray:
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0], [s,  c, 0], [0,  0, 1]])

def get_rotation_matrix_pitch_correction(deg: float) -> np.ndarray:
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])

def compute_move_xyz_from_measure(measure_res: MeasureDict, current_ry: float) -> Tuple[float, float, float]:
    mx = float(measure_res.get("move_x_mm", 0.0))
    my = float(measure_res.get("move_y_mm", 0.0))
    mz = float(measure_res.get("move_z_mm", 0.0))

    cx = mx + float(getattr(rc, "OFF_X_MM", 0.0))
    cy = my + float(getattr(rc, "OFF_Y_MM", 0.0))
    cz = mz + float(getattr(rc, "OFF_Z_MM", 0.0))

    vec_cam = np.array([-cx, cy, -cz])
    R_pitch = get_rotation_matrix_pitch_correction(current_ry)
    vec_horizontal = R_pitch @ vec_cam
    
    yaw_deg = float(getattr(rc, "BASE_YAW_OFFSET_DEG", -135.0))
    R_yaw = get_rotation_matrix_z(yaw_deg)
    vec_final = R_yaw @ vec_horizontal

    return float(vec_final[0]), float(vec_final[1]), float(vec_final[2])

def build_target_pose(current_tcp_pose6: Pose6, measure_res: MeasureDict) -> List[float]:
    """
    [UPGRADE] 측정된 기울기(tilt_ry)를 반영하여 Target Pose를 계산
    """
    x, y, z, rx, ry, rz = ensure_pose6(current_tcp_pose6)

    # 1. 이동량 계산
    dx, dy, dz = compute_move_xyz_from_measure(measure_res, ry)
    target_x = x + dx
    target_y = y + dy
    target_z = z + dz

    # 2. 손목 각도(Ry) 설정 (중요: 비전 측정값 반영)
    # 기본적으로 측정된 각도(tilt_ry)를 사용하되, 너무 과도하면 제한(Clamp)
    measured_ry = float(measure_res.get("tilt_ry", 2.0))
    
    # 안전장치: -30도 ~ +30도 사이로 제한 (하드웨어 보호)
    target_ry = max(-30.0, min(30.0, measured_ry))

    # [주의] 만약 로봇이 반대로 꺾인다면 아래 부호를 반대로 변경하세요 (-measured_ry)
    # target_ry = -target_ry 

    # 3. Pivot 보정 (가변 각도 target_ry에 맞춰 자동 계산)
    L = float(getattr(rc, "PIVOT_LENGTH", 165.0))
    
    rad_curr = np.deg2rad(ry)
    rad_targ = np.deg2rad(target_ry) # 2.0 대신 계산된 각도 사용

    comp_y_local = L * (np.sin(rad_curr) - np.sin(rad_targ))
    comp_z_local = L * (np.cos(rad_targ) - np.cos(rad_curr))

    pivot_vec = np.array([0, comp_y_local, 0])
    yaw_deg = float(getattr(rc, "BASE_YAW_OFFSET_DEG", -135.0))
    R_yaw = get_rotation_matrix_z(yaw_deg)
    pivot_rotated = R_yaw @ pivot_vec

    comp_dx = pivot_rotated[0]
    comp_dy = pivot_rotated[1]

    # 4. RZ (회전) 보정
    box_angle = float(measure_res.get("angle_deg", 0.0))
    target_rz = rz - box_angle

    # 최종 좌표
    final_x = target_x - comp_dx
    final_y = target_y - comp_dy
    final_z = target_z - comp_z_local

    return [final_x, final_y, final_z, rx, target_ry, target_rz]

def build_target_pose_with_debug(current_tcp_pose6: Pose6, measure_res: MeasureDict) -> Dict[str, object]:
    x, y, z, rx, ry, rz = ensure_pose6(current_tcp_pose6)
    target = build_target_pose(current_tcp_pose6, measure_res)
    
    return {
        "target_pose6": target,
        "move_x_mm": target[0] - x,
        "move_y_mm": target[1] - y,
        "move_z_mm": target[2] - z,
        "current_ry": ry,
        "measured_ry": float(measure_res.get("tilt_ry", 0.0)), # 디버깅용
        "target_ry": target[4],
        "angle_deg": float(measure_res.get("angle_deg", 0.0)),
    }

def fmt_pose6(pose6: Pose6) -> str:
    x, y, z, rx, ry, rz = ensure_pose6(pose6)
    return f"[x,y,z,rx,ry,rz]=[{x:.3f}, {y:.3f}, {z:.3f}, {rx:.3f}, {ry:.3f}, {rz:.3f}]"