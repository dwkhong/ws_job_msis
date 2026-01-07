# robot/target_pose.py
# ============================================================
# CMD 3: target_pose 생성 (1번 tcp_pose + 2번 measure(camXYZ, angle))
# - 로봇 통신 X (순수 계산)
# - "현재 TCP 각도(rz 등)"는 사용 안 함 (요청대로 제거)
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


def rot2d_xy(x: float, y: float, deg: float) -> Tuple[float, float]:
    rad = np.deg2rad(deg)
    c, s = float(np.cos(rad)), float(np.sin(rad))
    return (c * x - s * y), (s * x + c * y)


def compute_move_xyz_from_measure(measure_res: MeasureDict) -> Tuple[float, float, float]:
    """
    measure_res: {"cam_x_mm","cam_y_mm","cam_z_mm","angle_deg"(optional)}
    return: (move_x_mm, move_y_mm, move_z_mm)  # base에서 더할 delta
    """

    # 1) cam 측정값
    cam_x = float(measure_res["cam_x_mm"])
    cam_y = float(measure_res["cam_y_mm"])
    cam_z = float(measure_res["cam_z_mm"])

    # 2) cam -> gripper offset (robot_config에서)
    #    없으면 0으로 처리 (코드가 바로 터지지 않게)
    off_x = float(getattr(rc, "OFF_X_MM", 0.0))
    off_y = float(getattr(rc, "OFF_Y_MM", 0.0))
    off_z = float(getattr(rc, "OFF_Z_MM", 0.0))

    gx = cam_x + off_x
    gy = cam_y + off_y
    gz = cam_z + off_z

    # 3) 부호 매핑 (너가 기존 measure_box에서 쓰던 규칙 유지)
    dx0 = -gx
    dy0 = +gy
    dz0 = -gz

    # 4) 고정 yaw 보정: -135도 (또는 robot_config 값)
    yaw_deg = float(getattr(rc, "BASE_YAW_OFFSET_DEG", -135.0))
    dx1, dy1 = rot2d_xy(dx0, dy0, yaw_deg)

    # 5) flip (옵션)
    if bool(getattr(rc, "FLIP_MOVE_X", False)):
        dx1 = -dx1
    if bool(getattr(rc, "FLIP_MOVE_Y", False)):
        dy1 = -dy1

    return float(dx1), float(dy1), float(dz0)


def build_target_pose(current_tcp_pose6: Pose6, measure_res: MeasureDict) -> List[float]:
    """
    current_tcp_pose6: [x,y,z,rx,ry,rz] (mm, deg)
    measure_res: {"cam_x_mm","cam_y_mm","cam_z_mm","angle_deg"(optional)}
    return: target_pose6 = current + moveXYZ (orientation 유지)
    """
    x, y, z, rx, ry, rz = ensure_pose6(current_tcp_pose6)

    dx, dy, dz = compute_move_xyz_from_measure(measure_res)

    return [x + dx, y + dy, z + dz, rx, ry, rz]


def build_target_pose_with_debug(current_tcp_pose6: Pose6, measure_res: MeasureDict) -> Dict[str, object]:
    """
    디버그용: move_xyz도 같이 보고 싶을 때
    """
    dx, dy, dz = compute_move_xyz_from_measure(measure_res)
    target = build_target_pose(current_tcp_pose6, measure_res)
    return {
        "target_pose6": target,
        "move_x_mm": dx,
        "move_y_mm": dy,
        "move_z_mm": dz,
        "angle_deg": float(measure_res.get("angle_deg", 0.0)),
    }


def fmt_pose6(pose6: Pose6) -> str:
    x, y, z, rx, ry, rz = ensure_pose6(pose6)
    return f"[x,y,z,rx,ry,rz]=[{x:.3f}, {y:.3f}, {z:.3f}, {rx:.3f}, {ry:.3f}, {rz:.3f}]"








