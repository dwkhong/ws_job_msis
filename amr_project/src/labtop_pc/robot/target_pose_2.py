from __future__ import annotations
from typing import Dict, List, Sequence, Tuple, Union, Any
import numpy as np

# 기존 설정 로드
from . import robot_config as rc

Pose6 = Union[List[float], Tuple[float, float, float, float, float, float]]
MeasureDict = Dict[str, Union[int, float, str]]

def ensure_pose6(pose: Sequence[Union[int, float]]) -> List[float]:
    if not isinstance(pose, (list, tuple)) or len(pose) < 6:
        raise ValueError(f"pose6 must be list/tuple len>=6")
    return [float(x) for x in pose[:6]]

# =========================================================
# [UPGRADE] 행렬 연산 도우미 함수 (유지)
# =========================================================
def get_rotation_matrix_z(deg: float) -> np.ndarray:
    """ Z축(Yaw) 기준 회전 행렬 """
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

def get_rotation_matrix_pitch_correction(deg: float) -> np.ndarray:
    """ 로봇 Pitch(Ry) 보정용 회전 행렬 """
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [1, 0,  0],
        [0, c,  s],
        [0, -s, c]
    ])

# =========================================================
# 메인 계산 로직
# =========================================================
def compute_move_xyz_from_measure(measure_res: MeasureDict, current_ry: float) -> Tuple[float, float, float]:
    """
    1단계: 카메라 측정값 -> 로봇 이동량 변환 (행렬 연산 적용)
    """
    # 1) 측정값 가져오기 (호환성 확보)
    mx = float(measure_res.get("move_x_mm", measure_res.get("cam_x_mm", 0.0)))
    my = float(measure_res.get("move_y_mm", measure_res.get("cam_y_mm", 0.0)))
    mz = float(measure_res.get("move_z_mm", measure_res.get("cam_z_mm", 0.0)))

    cx = mx + float(getattr(rc, "OFF_X_MM", 0.0))
    cy = my + float(getattr(rc, "OFF_Y_MM", 0.0))
    cz = mz + float(getattr(rc, "OFF_Z_MM", 0.0))

    # 2) 카메라 로컬 벡터 (기존 부호 매핑 -x, y, -z 유지)
    vec_cam = np.array([-cx, cy, -cz])

    # 3) Pitch (Ry) 회전 행렬 적용
    R_pitch = get_rotation_matrix_pitch_correction(current_ry)
    vec_horizontal = R_pitch @ vec_cam

    # 4) Yaw (Base Offset) 회전 행렬 적용
    yaw_deg = float(getattr(rc, "BASE_YAW_OFFSET_DEG", -135.0))
    R_yaw = get_rotation_matrix_z(yaw_deg)
    
    # 최종 벡터
    vec_final = R_yaw @ vec_horizontal

    return float(vec_final[0]), float(vec_final[1]), float(vec_final[2])


def build_target_pose(current_tcp_pose6: Pose6, measure_res: MeasureDict) -> List[float]:
    """
    2단계: 최종 Target Pose 계산 (Pivot + Rz 보정 포함)
    ※ 접근 위치(Approach)는 cmd4/cmd7에서 내부적으로 처리하므로, 여기서는 최종 위치만 반환합니다.
    """
    x, y, z, rx, ry, rz = ensure_pose6(current_tcp_pose6)

    # [A] 기본 XYZ 이동량 계산
    dx, dy, dz = compute_move_xyz_from_measure(measure_res, ry)
    
    target_x = x + dx
    target_y = y + dy
    target_z = z + dz

    # [B] Pivot 보정 (팔 펴짐 보정 - 행렬/벡터 연산 활용)
    L = float(getattr(rc, "PIVOT_LENGTH", 165.0)) 
    target_ry = 2.0
    
    rad_curr = np.deg2rad(ry)
    rad_targ = np.deg2rad(target_ry)

    # 로컬 Y, Z 변위
    comp_y_local = L * (np.sin(rad_curr) - np.sin(rad_targ))
    comp_z_local = L * (np.cos(rad_targ) - np.cos(rad_curr))

    # 로컬 Y 변위를 베이스 Yaw에 맞춰 회전
    pivot_vec = np.array([0, comp_y_local, 0])
    yaw_deg = float(getattr(rc, "BASE_YAW_OFFSET_DEG", -135.0))
    R_yaw = get_rotation_matrix_z(yaw_deg)
    
    pivot_rotated = R_yaw @ pivot_vec

    comp_dx = pivot_rotated[0]
    comp_dy = pivot_rotated[1]

    # [C] RZ 각도 보정
    box_angle = float(measure_res.get("angle_deg", 0.0))
    target_rz = rz - box_angle

    # 최종 합산
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
    
    return {
        "target_pose6": target,
        "move_x_mm": target[0] - x,
        "move_y_mm": target[1] - y,
        "move_z_mm": target[2] - z,
        "current_ry": ry,
        "target_ry": target[4],
        "angle_deg": float(measure_res.get("angle_deg", 0.0)),
    }

def fmt_pose6(pose6: Pose6) -> str:
    x, y, z, rx, ry, rz = ensure_pose6(pose6)
    return f"[x,y,z,rx,ry,rz]=[{x:.3f}, {y:.3f}, {z:.3f}, {rx:.3f}, {ry:.3f}, {rz:.3f}]"