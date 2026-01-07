# robot/target_pose.py
# ============================================================
# CMD 3: target_pose 생성 (1번 tcp_pose + 2번 measure(camXYZ, angle))
# - 로봇 통신 X (순수 계산)
# - ✅ OFF_* / CAM_OFF_* 모두 지원 (키 불일치 방지)
# - ✅ (옵션) 현재 tcp ry(피치)로 X-Z 섞임 보정
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


def rot2d_xz(x: float, z: float, deg: float) -> Tuple[float, float]:
    """Y축(피치) 회전 효과: X-Z 평면에서 회전"""
    rad = np.deg2rad(deg)
    c, s = float(np.cos(rad)), float(np.sin(rad))
    # (x, z) -> (x', z')
    # ✅ camZ 성분이 X로 섞이도록(피치 보정) 설계
    xp = c * x + s * z
    zp = -s * x + c * z
    return xp, zp


def _get_cam_off_mm() -> Tuple[float, float, float]:
    """
    ✅ robot_config 키 불일치 대비
    - 우선순위: CAM_OFF_* -> OFF_*
    """
    def pick(name_a: str, name_b: str, default: float = 0.0) -> float:
        if hasattr(rc, name_a):
            return float(getattr(rc, name_a))
        if hasattr(rc, name_b):
            return float(getattr(rc, name_b))
        return float(default)

    off_x = pick("CAM_OFF_X_MM", "OFF_X_MM", 0.0)
    off_y = pick("CAM_OFF_Y_MM", "OFF_Y_MM", 0.0)
    off_z = pick("CAM_OFF_Z_MM", "OFF_Z_MM", 0.0)
    return off_x, off_y, off_z


def compute_move_xyz_from_measure(
    measure_res: MeasureDict,
    current_ry_deg: float = 0.0,
) -> Tuple[float, float, float, Dict[str, float]]:
    """
    measure_res: {"cam_x_mm","cam_y_mm","cam_z_mm","angle_deg"(optional)}
    return: (move_x_mm, move_y_mm, move_z_mm, dbg_dict)
    """

    # 1) cam 측정값
    cam_x = float(measure_res["cam_x_mm"])
    cam_y = float(measure_res["cam_y_mm"])
    cam_z = float(measure_res["cam_z_mm"])

    # 2) cam -> gripper offset
    off_x, off_y, off_z = _get_cam_off_mm()
    gx = cam_x + off_x
    gy = cam_y + off_y
    gz = cam_z + off_z

    # 3) 부호 매핑 (기존 규칙 유지)
    dx0 = -gx
    dy0 = +gy
    dz0 = -gz

    # 4) 고정 yaw 보정 (XY)
    yaw_deg = float(getattr(rc, "BASE_YAW_OFFSET_DEG", -135.0))
    dx1, dy1 = rot2d_xy(dx0, dy0, yaw_deg)

    # 5) (옵션) ry 피치 보정 (X-Z)
    #    ✅ 툴이 기울면 camZ(깊이)가 X로 섞여야 함
    apply_ry = bool(getattr(rc, "APPLY_TCP_RY_TO_MOVE", True))
    ry_sign  = float(getattr(rc, "TCP_RY_SIGN", +1.0))  # 필요시 부호 뒤집기
    ry_gain  = float(getattr(rc, "TCP_RY_GAIN", 1.0))   # 1.0이면 그대로, 0.8 등 튜닝 가능

    dx2, dz2 = dx1, dz0
    if apply_ry:
        dx2, dz2 = rot2d_xz(dx1, dz0, ry_sign * ry_gain * float(current_ry_deg))

    # 6) 스케일 튜닝(옵션)
    sxy = float(getattr(rc, "MOVE_XY_SCALE", 1.0))
    sz  = float(getattr(rc, "MOVE_Z_SCALE", 1.0))
    dx2 *= sxy
    dy1 *= sxy
    dz2 *= sz

    # 7) flip (옵션)
    if bool(getattr(rc, "FLIP_MOVE_X", False)):
        dx2 = -dx2
    if bool(getattr(rc, "FLIP_MOVE_Y", False)):
        dy1 = -dy1

    dbg = {
        "cam_x": cam_x, "cam_y": cam_y, "cam_z": cam_z,
        "off_x": off_x, "off_y": off_y, "off_z": off_z,
        "gx": gx, "gy": gy, "gz": gz,
        "dx0": dx0, "dy0": dy0, "dz0": dz0,
        "dx1": dx1, "dy1": dy1, "yaw_deg": yaw_deg,
        "apply_ry": float(apply_ry), "current_ry_deg": float(current_ry_deg),
        "dx_final": dx2, "dy_final": dy1, "dz_final": dz2,
    }

    return float(dx2), float(dy1), float(dz2), dbg


def build_target_pose(current_tcp_pose6: Pose6, measure_res: MeasureDict) -> List[float]:
    """
    current_tcp_pose6: [x,y,z,rx,ry,rz] (mm, deg)
    measure_res: {"cam_x_mm","cam_y_mm","cam_z_mm","angle_deg"(optional)}
    return: target_pose6 = current + moveXYZ (orientation 유지)
    """
    x, y, z, rx, ry, rz = ensure_pose6(current_tcp_pose6)
    dx, dy, dz, _ = compute_move_xyz_from_measure(measure_res, current_ry_deg=ry)
    return [x + dx, y + dy, z + dz, rx, ry, rz]


def build_target_pose_with_debug(current_tcp_pose6: Pose6, measure_res: MeasureDict) -> Dict[str, object]:
    """
    디버그용: 중간값까지 보기
    """
    x, y, z, rx, ry, rz = ensure_pose6(current_tcp_pose6)
    dx, dy, dz, dbg = compute_move_xyz_from_measure(measure_res, current_ry_deg=ry)
    target = [x + dx, y + dy, z + dz, rx, ry, rz]

    # ✅ phase0 비교용(원하면): Z hold 적용한 pose도 같이 제공
    z_hold = float(getattr(rc, "Z_HOLD_OFFSET_MM", 0.0))
    target_phase0 = [x + dx, y + dy, z + dz + z_hold, rx, ry, rz]

    return {
        "target_pose6": target,
        "target_phase0_pose6": target_phase0,
        "move_x_mm": dx,
        "move_y_mm": dy,
        "move_z_mm": dz,
        "angle_deg": float(measure_res.get("angle_deg", 0.0)),
        "dbg": dbg,
    }


def fmt_pose6(pose6: Pose6) -> str:
    x, y, z, rx, ry, rz = ensure_pose6(pose6)
    return f"[x,y,z,rx,ry,rz]=[{x:.3f}, {y:.3f}, {z:.3f}, {rx:.3f}, {ry:.3f}, {rz:.3f}]"



