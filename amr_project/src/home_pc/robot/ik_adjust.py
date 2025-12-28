# robot/ik_adjust.py
import time
import math

from . import robot_config as rc


def fmt_pose6(pose):
    if not isinstance(pose, (list, tuple)) or len(pose) < 6:
        return str(pose)
    x, y, z, rx, ry, rz = pose[:6]
    return f"[x,y,z,rx,ry,rz]=[{x:.3f}, {y:.3f}, {z:.3f}, {rx:.3f}, {ry:.3f}, {rz:.3f}]"


def ensure_pose6(p):
    if not isinstance(p, (list, tuple)) or len(p) < 6:
        raise ValueError(f"pose must be list/tuple len>=6, got={type(p)} len={len(p) if hasattr(p,'__len__') else 'N/A'}")
    return [float(x) for x in p[:6]]


def ensure_joint6(j):
    if not isinstance(j, (list, tuple)) or len(j) < 6:
        raise ValueError(f"joint must be list/tuple len>=6, got={type(j)} len={len(j) if hasattr(j,'__len__') else 'N/A'}")
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
                    try:
                        reconnect_cb()
                    except Exception:
                        pass
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
    if err != 0:
        return False
    return bool(ok)


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
    if err != 0:
        return None
    return ensure_joint6(j)


def _search_target_pose_only(robot, cur_joint6, base_target6, reconnect=None):
    """
    base_target6 주변에서 rx/ry/rz를 조금씩 바꿔서
    target_pose IK True 되는 후보를 찾음.
    """
    x, y, z, rx0, ry0, rz0 = ensure_pose6(base_target6)
    cur_joint6 = ensure_joint6(cur_joint6)

    rx_list = rc.SEARCH_RX_LIST
    ry_list = rc.SEARCH_RY_LIST
    rz_list = rc.SEARCH_RZ_LIST

    tries = 0
    t0 = time.time()
    best = None

    def timed_out():
        return (time.time() - t0) > float(rc.SEARCH_TIMEOUT_SEC)

    def check(cand_target, d_tuple):
        nonlocal tries
        tries += 1
        if timed_out() or tries > int(rc.SEARCH_MAX_TRIES):
            return None, True

        if not has_solution(robot, cand_target, cur_joint6, reconnect=reconnect):
            return None, False

        # score: joint 변화량 norm이 작은 후보 선호
        j_sol = get_ik(robot, cand_target, cur_joint6, reconnect=reconnect)
        if j_sol is None:
            return None, False

        score = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(j_sol, cur_joint6)))
        cand = {
            "target": ensure_pose6(cand_target),
            "score": float(score),
            "d": d_tuple,
            "tries": tries,
        }
        return cand, False

    # ✅ 우선순위: ry -> rz -> 전체(rpy)
    for dry in ry_list:
        cand_target = [x, y, z, rx0, ry0 + float(dry), rz0]
        cand, stop = check(cand_target, (0.0, float(dry), 0.0))
        if stop:
            return best
        if cand is not None:
            return cand

    for drz in rz_list:
        cand_target = [x, y, z, rx0, ry0, rz0 + float(drz)]
        cand, stop = check(cand_target, (0.0, 0.0, float(drz)))
        if stop:
            return best
        if cand is not None:
            return cand

    for drx in rx_list:
        for dry in ry_list:
            for drz in rz_list:
                cand_target = [x, y, z, rx0 + float(drx), ry0 + float(dry), rz0 + float(drz)]
                cand, stop = check(cand_target, (float(drx), float(dry), float(drz)))
                if stop:
                    return best
                if cand is None:
                    continue
                if (best is None) or (cand["score"] < best["score"]):
                    best = cand

    return best


def cmd4_check_and_adjust_target_only(robot, reconnect, cur_pose6, cur_joint6, target_pose6):
    """
    ✅ CMD4: target_pose만 IK 체크하고, 필요하면 RPY를 조금 바꿔 IK 되는 target_pose를 찾음.
    - phase0 관련 검증/보정은 '완전 무시'
    """
    try:
        target = ensure_pose6(target_pose6)
        cur_joint6 = ensure_joint6(cur_joint6)

        ok = has_solution(robot, target, cur_joint6, reconnect=reconnect)
        if ok:
            return {
                "ok": True,
                "adjusted": False,
                "msg": "IK OK (target only)",
                "target": target,
                "tries": 0,
                "score": 0.0,
                "d": (0.0, 0.0, 0.0),
            }

        best = _search_target_pose_only(
            robot=robot,
            cur_joint6=cur_joint6,
            base_target6=target,
            reconnect=reconnect,
        )

        if best is None:
            return {
                "ok": False,
                "adjusted": False,
                "msg": "IK failed: cannot find nearby solvable target_pose (target only search)",
                "target": target,
            }

        return {
            "ok": True,
            "adjusted": True,
            "msg": "IK adjusted (target only)",
            "target": best["target"],
            "tries": best["tries"],
            "score": best["score"],
            "d": best["d"],
        }

    except Exception as e:
        return {
            "ok": False,
            "adjusted": False,
            "msg": f"Exception in cmd4_check_and_adjust_target_only: {e}",
            "target": target_pose6,
        }
