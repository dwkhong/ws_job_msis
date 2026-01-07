from __future__ import annotations
from typing import Dict, List, Sequence, Tuple, Union, Optional
import time
import math
import numpy as np

# 사용자 설정 파일 (경로에 맞게 수정 필요)
from . import robot_config as rc

# --- 타입 정의 ---
Pose6 = Union[List[float], Tuple[float, float, float, float, float, float]]
Joint6 = Union[List[float], Tuple[float, float, float, float, float, float]]
MeasureDict = Dict[str, Union[int, float, str]]

# --- 기본 유틸리티 ---
def ensure_pose6(pose: Sequence[Union[int, float]]) -> List[float]:
    if not isinstance(pose, (list, tuple)) or len(pose) < 6:
        raise ValueError(f"pose6 must be list/tuple len>=6")
    return [float(x) for x in pose[:6]]

def ensure_joint6(j: Sequence[Union[int, float]]) -> List[float]:
    if not isinstance(j, (list, tuple)) or len(j) < 6:
        raise ValueError(f"joint must be list/tuple len>=6")
    return [float(x) for x in j[:6]]

def fmt_pose6(pose: Pose6) -> str:
    x, y, z, rx, ry, rz = ensure_pose6(pose)
    return f"[x,y,z,rx,ry,rz]=[{x:.3f}, {y:.3f}, {z:.3f}, {rx:.3f}, {ry:.3f}, {rz:.3f}]"

# --- [Core 1] 좌표 변환 및 타겟 계산 로직 ---

def compute_move_xyz_from_measure(measure_res: MeasureDict, current_ry: float) -> Tuple[float, float, float]:
    """
    카메라 측정값(상대좌표) -> 로봇 베이스 기준 이동량(절대좌표) 변환
    """
    # 1) cam 측정값 + 오프셋
    cx = float(measure_res["move_x_mm"]) + float(getattr(rc, "OFF_X_MM", 0.0))
    cy = float(measure_res["move_y_mm"]) + float(getattr(rc, "OFF_Y_MM", 0.0))
    cz = float(measure_res["move_z_mm"]) + float(getattr(rc, "OFF_Z_MM", 0.0))

    # 2) 카메라 좌표계 매핑 (사용자 환경에 맞게 부호 확인 필요)
    dx0 = -cx
    dy0 = cy
    dz0 = -cz

    # 3) Pitch (RY) 회전 보정 (카메라가 기울어진 상태 고려)
    rad_ry = np.deg2rad(current_ry)
    c, s = np.cos(rad_ry), np.sin(rad_ry)
    
    dx1 = dx0
    dy1 = dy0 * c + dz0 * s  # 수평 전진 성분
    dz1 = -dy0 * s + dz0 * c # 수직 하강 성분

    # 4) Yaw (Base Offset) 회전 보정 (로봇 베이스 설치 각도)
    yaw_deg = float(getattr(rc, "BASE_YAW_OFFSET_DEG", -135.0))
    rad_yaw = np.deg2rad(yaw_deg)
    c_y, s_y = np.cos(rad_yaw), np.sin(rad_yaw)
    
    dx_final = c_y * dx1 - s_y * dy1
    dy_final = s_y * dx1 + c_y * dy1
    dz_final = dz1

    return float(dx_final), float(dy_final), float(dz_final)


def build_target_pose(current_tcp_pose6: Pose6, measure_res: MeasureDict) -> List[float]:
    """
    최종 타겟 포즈 계산 (Pivot 보정 + RZ 회전 반영)
    """
    x, y, z, rx, ry, rz = ensure_pose6(current_tcp_pose6)

    # [A] 기본 이동량 계산
    dx, dy, dz = compute_move_xyz_from_measure(measure_res, ry)
    target_x = x + dx
    target_y = y + dy
    target_z = z + dz

    # [B] Pivot 보정: 손목을 펼 때(RY 변경) TCP가 밀려나는 현상 보정
    L = float(getattr(rc, "PIVOT_LENGTH", 165.0)) 
    target_ry = 2.0  # 최종적으로 잡을 때의 손목 각도 (거의 수직)
    
    rad_curr = np.deg2rad(ry)
    rad_targ = np.deg2rad(target_ry)

    # 로컬 변위(Y, Z) 계산
    comp_y_local = L * (np.sin(rad_curr) - np.sin(rad_targ))
    comp_z_local = L * (np.cos(rad_targ) - np.cos(rad_curr))

    # 로컬 변위 -> 월드 좌표(X, Y)로 분해
    yaw_deg = float(getattr(rc, "BASE_YAW_OFFSET_DEG", -135.0))
    rad_yaw = np.deg2rad(yaw_deg)
    
    comp_dx = -np.sin(rad_yaw) * comp_y_local
    comp_dy = np.cos(rad_yaw) * comp_y_local

    # [C] RZ 각도 보정 (OBB 박스 각도)
    box_angle = float(measure_res.get("angle_deg", 0.0))
    target_rz = rz - box_angle

    # [최종 합성]
    final_x = target_x - comp_dx
    final_y = target_y - comp_dy
    final_z = target_z - comp_z_local

    return [final_x, final_y, final_z, rx, target_ry, target_rz]


# --- [Core 2] IK 검증 및 최적해 탐색 (안전 진입 전략) ---

def safe_call(fn, *args, retry=1, sleep_sec=0.25, reconnect_cb=None, **kwargs):
    """로봇 통신 안전 호출 래퍼"""
    last_e = None
    for k in range(retry + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_e = e
            msg = str(e).lower()
            if ("timed out" in msg) or ("timeout" in msg) or ("실패" in str(e)):
                if reconnect_cb is not None:
                    try: reconnect_cb()
                    except: pass
            if k < retry:
                time.sleep(sleep_sec)
                continue
            raise last_e

def has_solution(robot, pose6, cur_joint6, reconnect=None) -> bool:
    pose6 = ensure_pose6(pose6)
    cur_joint6 = ensure_joint6(cur_joint6)
    try:
        err, ok = safe_call(
            robot.GetInverseKinHasSolution,
            int(rc.IK_REF),
            pose6,
            cur_joint6,
            retry=1,
            reconnect_cb=reconnect,
        )
        return (err == 0) and bool(ok)
    except:
        return False

def get_ik(robot, pose6, cur_joint6, reconnect=None) -> Optional[List[float]]:
    pose6 = ensure_pose6(pose6)
    cur_joint6 = ensure_joint6(cur_joint6)
    try:
        err, j = safe_call(
            robot.GetInverseKinRef,
            int(rc.IK_REF),
            pose6,
            cur_joint6,
            retry=1,
            reconnect_cb=reconnect,
        )
        return ensure_joint6(j) if err == 0 else None
    except:
        return None

def _search_target_and_phase0(robot, cur_joint6, base_target6, reconnect=None):
    """
    [안전 강화된 IK 탐색]
    1. Target(집기): 계산된 자세 그대로 가능해야 함.
    2. Phase0(진입): RZ(회전)는 Target과 같아야 함 (충돌 방지). 
                   단, RX/RY(기울기)는 수직(기본값)이어도 허용 (IK 해결).
    """
    x_base, y_base, z_base, rx_base, ry_base, rz_base = ensure_pose6(base_target6)
    cur_joint6 = ensure_joint6(cur_joint6)
    z_hold_offset = float(getattr(rc, "Z_HOLD_OFFSET_MM", 100.0))

    rx_list = rc.SEARCH_RX_LIST
    ry_list = rc.SEARCH_RY_LIST
    rz_list = rc.SEARCH_RZ_LIST

    tries = 0
    t0 = time.time()
    best = None

    def timed_out():
        return (time.time() - t0) > float(rc.SEARCH_TIMEOUT_SEC)

    def check_safe_descent(cand_target, d_tuple):
        nonlocal tries
        tries += 1
        if timed_out() or tries > int(rc.SEARCH_MAX_TRIES):
            return None, True

        # 1. [Target] 바닥 자세 검증 (필수)
        if not has_solution(robot, cand_target, cur_joint6, reconnect=reconnect):
            return None, False
        
        # 2. [Phase 0] 진입 자세 검증
        phase0_strict = list(cand_target)
        phase0_strict[2] += z_hold_offset # Z up
        
        # Case A: 완벽하게 동일한 자세로 진입 (Best)
        is_strict_ok = has_solution(robot, phase0_strict, cur_joint6, reconnect=reconnect)
        
        is_safe_ok = False
        final_phase0 = None

        if is_strict_ok:
            final_phase0 = phase0_strict
        else:
            # Case B: "회전(RZ)은 유지하되, 기울기(RX/RY)는 수직으로" (Safe Compromise)
            # -> 내려가면서 박스를 치지 않기 위해 RZ는 유지해야 함.
            phase0_safe = list(cand_target)
            phase0_safe[2] += z_hold_offset
            
            phase0_safe[3] = rx_base # 초기값(보통 180 or 0)
            phase0_safe[4] = ry_base # 초기값(보통 0)
            # phase0_safe[5] = cand_target[5] (RZ는 유지)
            
            is_safe_ok = has_solution(robot, phase0_safe, cur_joint6, reconnect=reconnect)
            if is_safe_ok:
                final_phase0 = phase0_safe

        if not (is_strict_ok or is_safe_ok):
            return None, False

        # 점수 계산 (현재 관절과 가까운지)
        j_sol = get_ik(robot, cand_target, cur_joint6, reconnect=reconnect)
        if j_sol is None: return None, False
        
        score = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(j_sol, cur_joint6)))
        
        return {
            "target": ensure_pose6(cand_target),
            "phase0": ensure_pose6(final_phase0), # 사용할 진입점 리턴
            "score": float(score),
            "d": d_tuple,
            "tries": tries,
            "mode": "STRICT" if is_strict_ok else "SAFE_TILT"
        }, False

    # --- 탐색 루프 (RZ우선 -> RX -> RY) ---
    
    # 1. RZ만 변경 (가장 빠르고 안전)
    for drz in rz_list:
        cand, stop = check_safe_descent(
            [x_base, y_base, z_base, rx_base, ry_base, rz_base + float(drz)], 
            (0.0, 0.0, float(drz))
        )
        if stop: return best
        if cand: return cand

    # 2. RX 변경
    for drx in rx_list:
        cand, stop = check_safe_descent(
            [x_base, y_base, z_base, rx_base + float(drx), ry_base, rz_base], 
            (float(drx), 0.0, 0.0)
        )
        if stop: return best
        if cand: return cand

    # 3. 전체 변경
    for dry in ry_list:
        for drx in rx_list:
            for drz in rz_list:
                cand, stop = check_safe_descent(
                    [x_base, y_base, z_base, rx_base + float(drx), ry_base + float(dry), rz_base + float(drz)], 
                    (float(drx), float(dry), float(drz))
                )
                if stop: return best
                if cand:
                    if (best is None) or (cand["score"] < best["score"]):
                        best = cand
    return best

# --- [메인 인터페이스] 외부에서 호출하는 함수 ---

def cmd4_check_and_adjust_target_only(robot, reconnect, cur_pose6: Pose6, cur_joint6: Joint6, target_pose6: Pose6) -> Dict:
    """
    1. 외부에서 계산된 target_pose6를 입력받음
    2. IK 검증 및 보정 (Safety Descent 적용)
    3. 결과 반환 (target, phase0)
    """
    try:
        # 1. 이미 main.py의 3번 메뉴에서 계산된 타겟을 사용
        base_target = ensure_pose6(target_pose6)
        
        # 2. IK 최적해 탐색 (phase0 포함)
        best = _search_target_and_phase0(robot, cur_joint6, base_target, reconnect=reconnect)

        if best is None:
            return {
                "ok": False,
                "msg": "IK 실패: 안전한 진입 경로를 찾을 수 없음",
                "target": base_target, # 실패시 원본 반환
            }

        # 성공
        return {
            "ok": True,
            "msg": f"IK 완료 (Mode: {best['mode']}, Tries: {best['tries']})",
            "target": best["target"],   # ✅ main.py가 기대하는 키 이름 ("target")
            "phase0": best["phase0"],   # ✅ main.py가 기대하는 키 이름 ("phase0")
            "score": best["score"],
            "adjusted": (best["mode"] == "SAFE_TILT"), # 보정 여부 플래그
            "d": best["d"]
        }

    except Exception as e:
        return {"ok": False, "msg": f"에러 발생: {e}"}