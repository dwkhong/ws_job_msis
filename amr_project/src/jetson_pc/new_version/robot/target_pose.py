# robot/target_pose.py
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union, Optional
import numpy as np

from . import robot_config as rc
from vision import measure_box_2 as mb
from . import robot_state as rs

Pose6 = Union[List[float], Tuple[float, float, float, float, float, float]]
MeasureDict = Dict[str, Union[int, float, str]]

# ============================================================
# ✅ Module cache (4번 결과를 5번에서 쓰기 위함)
# ============================================================
_last_target_pose6: Optional[List[float]] = None
_last_target_dbg: Optional[Dict[str, object]] = None


def set_last_target_pose6(pose6: Optional[Pose6], dbg: Optional[Dict[str, object]] = None) -> None:
    global _last_target_pose6, _last_target_dbg
    _last_target_pose6 = None if pose6 is None else ensure_pose6(pose6)
    _last_target_dbg = dbg


def get_last_target_pose6() -> Optional[List[float]]:
    return _last_target_pose6


def get_last_target_debug() -> Optional[Dict[str, object]]:
    return _last_target_dbg


# ============================================================
# Utils
# ============================================================
def ensure_pose6(pose: Sequence[Union[int, float]]) -> List[float]:
    if not isinstance(pose, (list, tuple)) or len(pose) < 6:
        raise ValueError("pose6 must be list/tuple len>=6")
    return [float(x) for x in pose[:6]]


def fmt_pose6(pose6: Pose6) -> str:
    x, y, z, rx, ry, rz = ensure_pose6(pose6)
    return f"[x,y,z,rx,ry,rz]=[{x:.3f}, {y:.3f}, {z:.3f}, {rx:.3f}, {ry:.3f}, {rz:.3f}]"


# ============================================================
# Core math
# ============================================================
def compute_move_xyz_from_measure(measure_res: MeasureDict, current_ry: float) -> Tuple[float, float, float]:
    cx = float(measure_res["move_x_mm"]) + float(getattr(rc, "OFF_X_MM", 0.0))
    cy = float(measure_res["move_y_mm"]) + float(getattr(rc, "OFF_Y_MM", 0.0))
    cz = float(measure_res["move_z_mm"]) + float(getattr(rc, "OFF_Z_MM", 0.0))

    dx0 = -cx
    dy0 = cy
    dz0 = -cz

    rad_ry = np.deg2rad(float(current_ry))
    c, s = float(np.cos(rad_ry)), float(np.sin(rad_ry))

    dx1 = dx0
    dy1 = dy0 * c + dz0 * s
    dz1 = -dy0 * s + dz0 * c

    yaw_deg = float(getattr(rc, "BASE_YAW_OFFSET_DEG", -135.0))
    rad_yaw = np.deg2rad(yaw_deg)
    c_y, s_y = float(np.cos(rad_yaw)), float(np.sin(rad_yaw))

    dx_final = c_y * dx1 - s_y * dy1
    dy_final = s_y * dx1 + c_y * dy1
    dz_final = dz1

    return float(dx_final), float(dy_final), float(dz_final)


def build_target_pose(current_tcp_pose6: Pose6, measure_res: MeasureDict) -> List[float]:
    x, y, z, rx, ry, rz = ensure_pose6(current_tcp_pose6)

    dx, dy, dz = compute_move_xyz_from_measure(measure_res, ry)
    target_x = x + dx
    target_y = y + dy
    target_z = z + dz

    L = float(getattr(rc, "PIVOT_LENGTH", 165.0))
    target_ry = float(getattr(rc, "TARGET_RY_DEG", 2.0))

    rad_curr = np.deg2rad(float(ry))
    rad_targ = np.deg2rad(float(target_ry))

    comp_y_local = L * (float(np.sin(rad_curr)) - float(np.sin(rad_targ)))
    comp_z_local = L * (float(np.cos(rad_targ)) - float(np.cos(rad_curr)))

    yaw_deg = float(getattr(rc, "BASE_YAW_OFFSET_DEG", -135.0))
    rad_yaw = np.deg2rad(yaw_deg)

    comp_dx = -float(np.sin(rad_yaw)) * comp_y_local
    comp_dy =  float(np.cos(rad_yaw)) * comp_y_local

    box_angle = float(measure_res.get("angle_deg", 0.0))
    target_rz = float(rz) - box_angle

    final_x = target_x - comp_dx
    final_y = target_y - comp_dy
    final_z = target_z - comp_z_local

    return [float(final_x), float(final_y), float(final_z), float(rx), float(target_ry), float(target_rz)]


def build_target_pose_with_debug(current_tcp_pose6: Pose6, measure_res: MeasureDict) -> Dict[str, object]:
    x, y, z, rx, ry, rz = ensure_pose6(current_tcp_pose6)
    target = build_target_pose(current_tcp_pose6, measure_res)

    return {
        "target_pose6": target,
        "move_x_mm": float(target[0] - x),
        "move_y_mm": float(target[1] - y),
        "move_z_mm": float(target[2] - z),
        "current_ry": float(ry),
        "target_ry": float(target[4]),
        "angle_deg": float(measure_res.get("angle_deg", 0.0)),
        "cur_pose6": [float(x), float(y), float(z), float(rx), float(ry), float(rz)],
        "measure": dict(measure_res),
    }


# ============================================================
# Commands (4번: 캐시만 저장)
# ============================================================
def cmd_build_target_from_last(robot=None, reconnect=None, use_last_pose: bool = True) -> Optional[Dict[str, object]]:
    # lazy import (순환 import 방지)
   

    meas = mb.get_last_measure_avg() if hasattr(mb, "get_last_measure_avg") else None
    if meas is None:
        set_last_target_pose6(None, None)
        return None

    pose = None
    if use_last_pose:
        if hasattr(rs, "get_last_pose_joint"):
            pose, _ = rs.get_last_pose_joint()
        if pose is None:
            set_last_target_pose6(None, None)
            return None
    else:
        if robot is None:
            set_last_target_pose6(None, None)
            return None
        (err_p, pose), _ = rs.read_pose_joint(robot, reconnect=reconnect)
        if err_p != 0 or pose is None:
            set_last_target_pose6(None, None)
            return None

    dbg = build_target_pose_with_debug(pose, meas)
    set_last_target_pose6(dbg["target_pose6"], dbg)
    return dbg
