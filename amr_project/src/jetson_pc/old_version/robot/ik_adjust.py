import time
import math
import numpy as np
from . import robot_config as rc

def fmt_pose6(pose):
    if not isinstance(pose, (list, tuple)) or len(pose) < 6:
        return str(pose)
    x, y, z, rx, ry, rz = pose[:6]
    return f"[x,y,z,rx,ry,rz]=[{x:.3f}, {y:.3f}, {z:.3f}, {rx:.3f}, {ry:.3f}, {rz:.3f}]"

def ensure_pose6(p):
    if not isinstance(p, (list, tuple)) or len(p) < 6:
        raise ValueError(f"pose must be list/tuple len>=6")
    return [float(x) for x in p[:6]]

def ensure_joint6(j):
    if not isinstance(j, (list, tuple)) or len(j) < 6:
        raise ValueError(f"joint must be list/tuple len>=6")
    return [float(x) for x in j[:6]]

def safe_call(fn, *args, retry=1, sleep_sec=0.25, reconnect_cb=None, **kwargs):
    last_e = None
    for k in range(retry + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_e = e
            msg = str(e).lower()
            if ("timed out" in msg) or ("timeout" in msg) or ("实时数据失败" in str(e)):
                if reconnect_cb is not None:
                    try: reconnect_cb()
                    except: pass
            if k < retry:
                time.sleep(sleep_sec)
                continue
            raise last_e

def has_solution(robot, pose6, cur_joint6, reconnect=None):
    pose6 = ensure_pose6(pose6)
    cur_joint6 = ensure_joint6(cur_joint6)
    err, ok = safe_call(
        robot.GetInverseKinHasSolution,
        int(rc.IK_REF),
        pose6,
        cur_joint6,
        retry=1,
        reconnect_cb=reconnect,
    )
    return (err == 0) and bool(ok)

def get_ik(robot, pose6, cur_joint6, reconnect=None):
    pose6 = ensure_pose6(pose6)
    cur_joint6 = ensure_joint6(cur_joint6)
    err, j = safe_call(
        robot.GetInverseKinRef,
        int(rc.IK_REF),
        pose6,
        cur_joint6,
        retry=1,
        reconnect_cb=reconnect,
    )
    return ensure_joint6(j) if err == 0 else None

def _search_target_and_phase0(robot, cur_joint6, base_target6, reconnect=None):
    """
    수정된 우선순위: RZ(회전) -> RX(비틀기) -> RY(손목각도) 순서로 탐색
    """
    x, y, z, rx0, ry0, rz0 = ensure_pose6(base_target6)
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

    def check_both(cand_target, d_tuple):
        nonlocal tries
        tries += 1
        if timed_out() or tries > int(rc.SEARCH_MAX_TRIES):
            return None, True

        if not has_solution(robot, cand_target, cur_joint6, reconnect=reconnect):
            return None, False
        
        phase0 = list(cand_target)
        phase0[2] += z_hold_offset
        if not has_solution(robot, phase0, cur_joint6, reconnect=reconnect):
            return None, False

        j_sol = get_ik(robot, cand_target, cur_joint6, reconnect=reconnect)
        if j_sol is None:
            return None, False

        score = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(j_sol, cur_joint6)))
        return {
            "target": ensure_pose6(cand_target),
            "score": float(score),
            "d": d_tuple,
            "tries": tries,
        }, False

    # --- 우선순위 탐색 시작 (순서 변경됨) ---

    # 1순위: RZ (그리퍼 자체 회전)만 변경
    for drz in rz_list:
        cand, stop = check_both([x, y, z, rx0, ry0, rz0 + float(drz)], (0.0, 0.0, float(drz)))
        if stop: return best
        if cand: return cand

    # 2순위: RX (좌우 기울기)만 변경
    for drx in rx_list:
        cand, stop = check_both([x, y, z, rx0 + float(drx), ry0, rz0], (float(drx), 0.0, 0.0))
        if stop: return best
        if cand: return cand

    # 3순위: RY 및 전체 조합 (가장 마지막에 시도)
    for dry in ry_list:
        for drx in rx_list:
            for drz in rz_list:
                # 0.0, 0.0, 0.0은 이미 위에서 걸러졌거나 첫 시도이므로 건너뛰어도 됨
                cand, stop = check_both([x, y, z, rx0 + float(drx), ry0 + float(dry), rz0 + float(drz)], 
                                       (float(drx), float(dry), float(drz)))
                if stop: return best
                if cand:
                    if (best is None) or (cand["score"] < best["score"]):
                        best = cand
    return best

def cmd4_check_and_adjust_target_only(robot, reconnect, cur_pose6, cur_joint6, target_pose6):
    """
    ✅ 수정된 CMD4: target + phase0 세트로 IK를 검증하고 보정함
    """
    try:
        target = ensure_pose6(target_pose6)
        cur_joint6 = ensure_joint6(cur_joint6)
        z_hold_offset = float(getattr(rc, "Z_HOLD_OFFSET_MM", 100.0))
        
        # 1. 현재 타겟의 바닥과 공중을 모두 확인
        phase0_base = target[:]
        phase0_base[2] += z_hold_offset
        
        ok_target = has_solution(robot, target, cur_joint6, reconnect=reconnect)
        ok_phase0 = has_solution(robot, phase0_base, cur_joint6, reconnect=reconnect)

        if ok_target and ok_phase0:
            return {
                "ok": True,
                "adjusted": False,
                "msg": "이미 IK OK (target + phase0 모두 가능)",
                "target": target,
                "tries": 0, "score": 0.0, "d": (0.0, 0.0, 0.0),
            }

        # 2. 둘 중 하나라도 안 되면 통합 검색 시작
        print(f"[CMD4] IK 부족 (Target:{ok_target}, Phase0:{ok_phase0}) -> 통합 보정 시작...")
        best = _search_target_and_phase0(robot, cur_joint6, target, reconnect=reconnect)

        if best is None:
            return {
                "ok": False,
                "adjusted": False,
                "msg": "IK 실패: 바닥과 공중을 동시에 만족하는 자세를 찾을 수 없음",
                "target": target,
            }

        return {
            "ok": True,
            "adjusted": True,
            "msg": f"IK 보정 완료 (target+phase0 세트)",
            "target": best["target"],
            "tries": best["tries"],
            "score": best["score"],
            "d": best["d"],
        }

    except Exception as e:
        return {"ok": False, "msg": f"CMD4 에러: {e}", "target": target_pose6}