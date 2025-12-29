# robot/target_pose.py
# ============================================================
# CMD 3: target_pose 생성 (1번 tcp_pose + 2번 measure(moveXYZ))
# - 로봇 통신 X (순수 계산)
# ============================================================

from __future__ import annotations
from typing import Dict, List, Sequence, Tuple, Union


Pose6 = Union[List[float], Tuple[float, float, float, float, float, float]]
MeasureDict = Dict[str, Union[int, float, str]]


def ensure_pose6(pose: Sequence[Union[int, float]]) -> List[float]:
    if not isinstance(pose, (list, tuple)) or len(pose) < 6:
        raise ValueError(
            f"pose6 must be list/tuple len>=6, got: {type(pose)} "
            f"len={len(pose) if hasattr(pose,'__len__') else 'N/A'}"
        )
    return [float(x) for x in pose[:6]]


def build_target_pose(current_tcp_pose6: Pose6, measure_res: MeasureDict) -> List[float]:
    """
    current_tcp_pose6: [x,y,z,rx,ry,rz] (mm, deg)
    measure_res: {"move_x_mm","move_y_mm","move_z_mm", ...}
    return: target_pose6 = current + moveXYZ (orientation 유지)
    """
    x, y, z, rx, ry, rz = ensure_pose6(current_tcp_pose6)

    dx = float(measure_res["move_x_mm"])
    dy = float(measure_res["move_y_mm"])
    dz = float(measure_res["move_z_mm"])

    return [x + dx, y + dy, z + dz, rx, ry, rz]


def fmt_pose6(pose6: Pose6) -> str:
    p = ensure_pose6(pose6)
    x, y, z, rx, ry, rz = p
    return f"[x,y,z,rx,ry,rz]=[{x:.3f}, {y:.3f}, {z:.3f}, {rx:.3f}, {ry:.3f}, {rz:.3f}]"
