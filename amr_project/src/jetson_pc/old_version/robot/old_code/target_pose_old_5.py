# robot/target_pose.py
# ============================================================
# 보정 사항: RY(Pitch) 각도에 따른 3D 좌표 변환 추가
# 로봇 팔이 숙여져 있을 때 카메라의 Y축 이동이 
# 실제 Base 기준 Y와 Z로 분산되는 현상을 계산함
# ============================================================

from __future__ import annotations
from typing import Dict, List, Sequence, Tuple, Union
import numpy as np

from . import robot_config as rc


Pose6 = Union[List[float], Tuple[float, float, float, float, float, float]]
MeasureDict = Dict[str, Union[int, float, str]]


def ensure_pose6(pose: Sequence[Union[int, float]]) -> List[float]:
    if not isinstance(pose, (list, tuple)) or len(pose) < 6:
        raise ValueError(
            f"pose6 must be list/tuple len>=6, got: {type(pose)} "
            f"len={len(pose) if hasattr(pose,'__len__') else 'N/A'}"
        )
    return [float(x) for x in pose[:6]]


def compute_move_xyz_from_measure(measure_res: MeasureDict, current_ry: float) -> Tuple[float, float, float]:
    # 1) 카메라 측정값 (Offset 포함)
    # 카메라가 보는 타겟까지의 거리(깊이)가 cam_z
    cx = float(measure_res["cam_x_mm"]) + float(getattr(rc, "OFF_X_MM", 0.0))
    cy = float(measure_res["cam_y_mm"]) + float(getattr(rc, "OFF_Y_MM", 0.0))
    cz = float(measure_res["cam_z_mm"]) + float(getattr(rc, "OFF_Z_MM", 0.0))

    # 2) 카메라 로컬 벡터 설정 (영상 기반 부호 정의)
    # 영상에서 오른쪽(+)이 로봇 -X, 영상에서 위(+)가 로봇 +Y라고 가정 (데이터 기반 추론)
    dx0 = -cx  
    dy0 = cy   
    dz0 = -cz  # 타겟이 카메라 앞에 있으므로 - 방향 (로봇이 다가가야 함)

    # 3) 3D Pitch(RY) 회전 보정 (핵심)
    # 로봇의 RY가 숙여질수록(RY > 0), 카메라의 '앞'은 로봇의 '전진 + 하강'이 됨
    rad_ry = np.deg2rad(current_ry)
    c, s = np.cos(rad_ry), np.sin(rad_ry)

    # Pitch 회전 (X축 기준 회전)
    # 로봇 베이스 수평 기준의 델타값으로 변환
    dx1 = dx0
    # dy1: 로봇이 수평으로 전진해야 할 거리
    # dz1: 로봇이 수직으로 내려가야 할 거리
    dy1 = dy0 * c + dz0 * s
    dz1 = -dy0 * s + dz0 * c

    # 4) Yaw (-135도) 보정
    yaw_deg = float(getattr(rc, "BASE_YAW_OFFSET_DEG", -135.0))
    rad_yaw = np.deg2rad(yaw_deg)
    cy_y, sy_y = np.cos(rad_yaw), np.sin(rad_yaw)

    dx_final = cy_y * dx1 - sy_y * dy1
    dy_final = sy_y * dx1 + cy_y * dy1
    dz_final = dz1

    # 5) 수동 부호 조정 (데이터에서 Y가 -124로 가야하는데 -147로 갔으므로 반전 필요)
    # 현재 데이터 흐름상 dy_final의 방향이 반대인 것으로 보임
    return float(dx_final), float(dy_final), float(dz_final)


def build_target_pose(current_tcp_pose6: Pose6, measure_res: MeasureDict) -> List[float]:
    """
    current_tcp_pose6: [x,y,z,rx,ry,rz]
    return: 보정된 target_pose6
    """
    x, y, z, rx, ry, rz = ensure_pose6(current_tcp_pose6)

    # ry 값을 전달하여 3D 보정 계산
    dx, dy, dz = compute_move_xyz_from_measure(measure_res, ry)

    return [x + dx, y + dy, z + dz, rx, ry, rz]


def build_target_pose_with_debug(current_tcp_pose6: Pose6, measure_res: MeasureDict) -> Dict[str, object]:
    """
    디버그용: 계산 과정 포함
    """
    x, y, z, rx, ry, rz = ensure_pose6(current_tcp_pose6)
    dx, dy, dz = compute_move_xyz_from_measure(measure_res, ry)
    target = [x + dx, y + dy, z + dz, rx, ry, rz]
    
    return {
        "target_pose6": target,
        "move_x_mm": dx,
        "move_y_mm": dy,
        "move_z_mm": dz,
        "current_ry": ry,
        "angle_deg": float(measure_res.get("angle_deg", 0.0)),
    }


def fmt_pose6(pose6: Pose6) -> str:
    x, y, z, rx, ry, rz = ensure_pose6(pose6)
    return f"[x,y,z,rx,ry,rz]=[{x:.3f}, {y:.3f}, {z:.3f}, {rx:.3f}, {ry:.3f}, {rz:.3f}]"








