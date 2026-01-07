# robot/ik_check.py
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from . import robot_config as rc

Pose6 = Union[List[float], Tuple[float, float, float, float, float, float]]
Joint6 = Union[List[float], Tuple[float, float, float, float, float, float]]

_LAST_IK_RESULT: Optional[Dict[str, Any]] = None


# ============================================================
# utils
# ============================================================
def _ensure_pose6(p: Sequence[Union[int, float]]) -> List[float]:
    if not isinstance(p, (list, tuple)) or len(p) < 6:
        raise ValueError("pose6 must be list/tuple len>=6")
    return [float(x) for x in p[:6]]


def _ensure_joint6(j: Sequence[Union[int, float]]) -> List[float]:
    if not isinstance(j, (list, tuple)) or len(j) < 6:
        raise ValueError("joint6 must be list/tuple len>=6")
    return [float(x) for x in j[:6]]


def _is_timeout_like(e: Exception) -> bool:
    s = str(e).lower()
    return ("timeout" in s) or ("timed out" in s) or ("实时数据失败" in str(e))


def _safe_call(fn, *args, reconnect=None, **kwargs):
    last_e = None
    retry = int(getattr(rc, "RPC_RETRY", 1))
    sleep_sec = float(getattr(rc, "RPC_RETRY_SLEEP_SEC", 0.25))

    for k in range(retry + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_e = e
            # timeout류면 재연결 시도
            if _is_timeout_like(e) and reconnect is not None:
                try:
                    reconnect()
                except Exception:
                    pass
            if k < retry:
                time.sleep(sleep_sec)
                continue
            raise last_e


def _has_solution(robot, pose6: Pose6, cur_joint6: Joint6, reconnect=None) -> bool:
    pose6 = _ensure_pose6(pose6)
    cur_joint6 = _ensure_joint6(cur_joint6)
    err, ok = _safe_call(
        robot.GetInverseKinHasSolution,
        int(getattr(rc, "IK_REF", 0)),
        pose6,
        cur_joint6,
        reconnect=reconnect,
    )
    return (int(err) == 0) and bool(ok)


def fmt_pose6(pose6: Pose6) -> str:
    x, y, z, rx, ry, rz = _ensure_pose6(pose6)
    return f"[x,y,z,rx,ry,rz]=[{x:.3f}, {y:.3f}, {z:.3f}, {rx:.3f}, {ry:.3f}, {rz:.3f}]"


# ============================================================
# public cache getters
# ============================================================
def get_last_ik_result() -> Optional[Dict[str, Any]]:
    return _LAST_IK_RESULT


def get_last_phase0_pose6() -> Optional[List[float]]:
    if _LAST_IK_RESULT and _LAST_IK_RESULT.get("ok"):
        return _LAST_IK_RESULT.get("phase0_pose6")
    return None


# ============================================================
# main API
# ============================================================
def check_target_ik(
    robot,
    reconnect,
    cur_joint6: Joint6,
    target_pose6: Pose6,
    check_phase0: bool = True,
    z_hold_offset_mm: Optional[float] = None,
) -> Dict[str, Any]:
    """
    ✅ "타겟은 건드리지 않음"
    - target_pose6: 4번에서 계산된 목표 포즈
    - cur_joint6   : 2번에서 읽어온 현재 관절(캐시)
    """
    global _LAST_IK_RESULT

    if robot is None:
        out = {"ok": False, "msg": "robot is None"}
        _LAST_IK_RESULT = out
        return out

    cur_joint6 = _ensure_joint6(cur_joint6)
    target = _ensure_pose6(target_pose6)

    # 0) target IK
    try:
        target_ok = _has_solution(robot, target, cur_joint6, reconnect=reconnect)
    except Exception as e:
        out = {"ok": False, "msg": f"target IK check exception: {e}", "target_pose6": target}
        _LAST_IK_RESULT = out
        return out

    if not target_ok:
        out = {
            "ok": False,
            "target_ok": False,
            "phase0_ok": False,
            "target_pose6": target,
            "phase0_pose6": None,
            "mode": "TARGET_FAIL",
            "tries": 1,
            "msg": "target IK not solvable",
        }
        _LAST_IK_RESULT = out
        return out

    if not check_phase0:
        out = {
            "ok": True,
            "target_ok": True,
            "phase0_ok": True,
            "target_pose6": target,
            "phase0_pose6": None,
            "mode": "NO_PHASE0",
            "tries": 1,
            "msg": "target OK (phase0 skipped)",
        }
        _LAST_IK_RESULT = out
        return out

    zoff = float(getattr(rc, "Z_HOLD_OFFSET_MM", 70.0) if z_hold_offset_mm is None else z_hold_offset_mm)

    # 1) phase0 strict: 타겟 그대로 + Z만 위로
    phase0_strict = list(target)
    phase0_strict[2] = float(phase0_strict[2] + zoff)

    tries = 0
    try:
        tries += 1
        if _has_solution(robot, phase0_strict, cur_joint6, reconnect=reconnect):
            out = {
                "ok": True,
                "target_ok": True,
                "phase0_ok": True,
                "target_pose6": target,
                "phase0_pose6": phase0_strict,
                "mode": "STRICT",
                "tries": tries,
                "msg": "target OK, phase0 OK (strict)",
            }
            _LAST_IK_RESULT = out
            return out
    except Exception as e:
        out = {"ok": False, "msg": f"phase0 strict IK exception: {e}", "target_pose6": target}
        _LAST_IK_RESULT = out
        return out

    # 2) strict가 안 되면: phase0에서만 RX/RY를 살짝 스윕 (RZ는 유지)
    rx_list = list(getattr(rc, "SEARCH_RX_LIST", [0, 1, -1, 2, -2, 3, -3, 5, -5]))
    ry_list = list(getattr(rc, "SEARCH_RY_LIST", [0, 1, -1, 2, -2, 3, -3, 5, -5]))

    base_rx = float(target[3])
    base_ry = float(target[4])
    base_rz = float(target[5])

    timeout_sec = float(getattr(rc, "SEARCH_TIMEOUT_SEC", 6.0))
    max_tries = int(getattr(rc, "SEARCH_MAX_TRIES", 900))
    t0 = time.time()

    def timed_out() -> bool:
        return (time.time() - t0) > timeout_sec

    best: Optional[List[float]] = None

    for dry in ry_list:
        for drx in rx_list:
            tries += 1
            if tries > max_tries or timed_out():
                break

            cand = list(phase0_strict)
            cand[3] = base_rx + float(drx)
            cand[4] = base_ry + float(dry)
            cand[5] = base_rz  # ✅ RZ 유지

            try:
                if _has_solution(robot, cand, cur_joint6, reconnect=reconnect):
                    best = cand
                    break
            except Exception:
                continue

        if best is not None or tries > max_tries or timed_out():
            break

    if best is None:
        out = {
            "ok": False,
            "target_ok": True,
            "phase0_ok": False,
            "target_pose6": target,
            "phase0_pose6": None,
            "mode": "PHASE0_FAIL",
            "tries": tries,
            "msg": "target OK but phase0 IK not solvable",
        }
        _LAST_IK_RESULT = out
        return out

    out = {
        "ok": True,
        "target_ok": True,
        "phase0_ok": True,
        "target_pose6": target,
        "phase0_pose6": best,
        "mode": "SEARCH_TILT",
        "tries": tries,
        "msg": "target OK, phase0 OK (rx/ry sweep)",
    }
    _LAST_IK_RESULT = out
    return out


# ============================================================
# cmd for main menu (5번)
# ============================================================
def cmd_check_target_from_last(robot=None, reconnect=None, check_phase0: bool = True):
    """
    main 5번:
      - last_joint: robot_state(2번) 캐시
      - last_target: target_pose(4번) 캐시
    """
    global _LAST_IK_RESULT

    if robot is None:
        print("[IK] robot is None (0번으로 연결)")
        _LAST_IK_RESULT = {"ok": False, "msg": "robot is None"}
        return _LAST_IK_RESULT

    from . import robot_state as rs
    from . import target_pose as tp

    # joint 캐시
    if hasattr(rs, "get_last_joint6"):
        last_joint = rs.get_last_joint6()
    else:
        # 구버전 대비
        _, last_joint = rs.get_last_pose_joint() if hasattr(rs, "get_last_pose_joint") else (None, None)

    # target 캐시
    last_target = tp.get_last_target_pose6() if hasattr(tp, "get_last_target_pose6") else None

    if last_joint is None:
        print("[IK] last_joint6 is None (2번 먼저)")
        _LAST_IK_RESULT = {"ok": False, "msg": "last_joint6 is None"}
        return _LAST_IK_RESULT

    if last_target is None:
        print("[IK] last_target_pose6 is None (4번 먼저)")
        _LAST_IK_RESULT = {"ok": False, "msg": "last_target_pose6 is None"}
        return _LAST_IK_RESULT

    res = check_target_ik(
        robot,
        reconnect=reconnect,
        cur_joint6=last_joint,
        target_pose6=last_target,
        check_phase0=check_phase0,
    )

    # 최소 출력(원하면 여기 출력 더 줄여도 됨)
    if res.get("ok"):
        print(f"[IK] OK  mode={res.get('mode')}  tries={res.get('tries')}")
        print("[IK] target :", fmt_pose6(res["target_pose6"]))
        if res.get("phase0_pose6") is not None:
            print("[IK] phase0 :", fmt_pose6(res["phase0_pose6"]))
    else:
        print(f"[IK] FAIL: {res.get('msg')}")
        if res.get("target_pose6") is not None:
            print("[IK] target :", fmt_pose6(res["target_pose6"]))

    return res


