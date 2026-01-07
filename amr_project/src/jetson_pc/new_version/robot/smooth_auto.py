# robot/smooth_auto.py
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from . import robot_config as rc
from . import robot_state as rs
from . import gripper_control as gc

Pose6 = Union[List[float], Tuple[float, float, float, float, float, float]]
Joint6 = Union[List[float], Tuple[float, float, float, float, float, float]]


# ✅ cmd7(2-step)용 컨텍스트
_CMD7_CTX: Dict[str, Any] = {
    "armed": False,       # phase0까지 갔고, 다음 누르면 내려갈 준비됨
    "target": None,       # 당시의 target 캐시
    "phase0": None,       # 당시의 phase0 캐시
    "t": 0.0,             # armed 설정 시각
}


# ============================================================
# Utils
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
    # Fairino 쪽에서 종종 나오는 문자열도 포함
    return ("timeout" in s) or ("timed out" in s) or ("实时数据失败" in str(e))


def _safe_call(fn, *args, reconnect=None, **kwargs):
    """
    RPC 호출 재시도 래퍼
    - 정상 시 오버헤드 거의 없음
    - 실패 시에만 retry + sleep + (선택) reconnect
    """
    last_e = None
    retry = int(getattr(rc, "RPC_RETRY", 1))
    sleep_sec = float(getattr(rc, "RPC_RETRY_SLEEP_SEC", 0.25))

    for k in range(retry + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_e = e

            # timeout-like면 reconnect 시도(요청한 경우)
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
    """IK(역기구학) 해 존재 여부 확인"""
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


def _movecart_direct(
    robot,
    pose6: Pose6,
    tool: int,
    user: int,
    vel_list: Sequence[Union[int, float]],
    reconnect=None,
    label: str = "",
) -> int:
    """
    MoveCart 실행 (속도 fallback 포함)
    - vel_list 중 하나라도 성공하면 OK
    - rtn==112면 속도/조건 문제로 다음 속도 시도
    """
    pose6 = _ensure_pose6(pose6)

    acc = float(getattr(rc, "MOVE_CART_ACC", 0.0))
    ovl = float(getattr(rc, "MOVE_CART_OVL", 100.0))
    blendT = float(getattr(rc, "MOVE_CART_BLENDT", -1.0))
    config = int(getattr(rc, "MOVE_CART_CONFIG", -1))

    for vv in list(vel_list):
        rtn = _safe_call(
            robot.MoveCart,
            pose6,
            int(tool),
            int(user),
            float(vv),
            float(acc),
            float(ovl),
            float(blendT),
            int(config),
            reconnect=reconnect,
        )
        if int(rtn) == 0:
            if label:
                print(f"[MoveCart-OK] {label} vel={vv}")
            return 0

        # 112: 속도/제약 조건으로 실패할 때가 많아서 다음 속도 시도
        if int(rtn) == 112:
            if label:
                print(f"[MoveCart-112] {label} vel={vv} -> try next")
            continue

        if label:
            print(f"[MoveCart-FAIL] {label} vel={vv} err={rtn} -> try next")

    return 112


def _same_pose6(a: Optional[Pose6], b: Optional[Pose6], tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return False
    aa = _ensure_pose6(a)
    bb = _ensure_pose6(b)
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(aa, bb))


def cmd7_reset() -> None:
    _CMD7_CTX["armed"] = False
    _CMD7_CTX["target"] = None
    _CMD7_CTX["phase0"] = None
    _CMD7_CTX["t"] = 0.0


# ============================================================
# Core: Phase0 -> Target (Smooth Auto)
# ============================================================
def cmd_smooth_auto(
    robot,
    reconnect,
    target_pose6: Pose6,
    phase0_pose6: Pose6,
    state: Optional[Dict[str, Any]] = None,
    tool: int = 0,
    user: int = 0,
    auto_grip_close: bool = True,
    vel_list: Optional[Sequence[Union[int, float]]] = None,
) -> Dict[str, Any]:
    """
    ✅ 자동 접근(phase0) -> 하강(target) -> (옵션) 그리퍼 닫기
    - target_pose6 : 4번에서 계산된 최종 타겟
    - phase0_pose6 : 5번(ik_check)에서 계산된 안전 진입점
    """
    if robot is None:
        return {"ok": False, "msg": "robot is None"}

    if target_pose6 is None or phase0_pose6 is None:
        return {"ok": False, "msg": "target/phase0 is None"}

    target = _ensure_pose6(target_pose6)
    phase0 = _ensure_pose6(phase0_pose6)

    vel_list = list(getattr(rc, "MOVE_CART_VEL_LIST", [20, 10, 5])) if vel_list is None else list(vel_list)
    state = gc.get_state() if state is None else state

    # 0) 현재 상태 읽기
    (e1, _cur_pose6), (e2, cur_joint6) = rs.read_pose_joint(robot, reconnect=reconnect)
    if e1 != 0 or e2 != 0 or cur_joint6 is None:
        return {"ok": False, "msg": f"상태 읽기 실패 err_p={e1}, err_j={e2}"}

    # 1) phase0 IK 체크 + 이동
    if not _has_solution(robot, phase0, cur_joint6, reconnect=reconnect):
        return {"ok": False, "msg": "Phase0 IK 불가(이동 불가)", "phase0": phase0}

    print("\n[SMOOTH] 1) Phase0 MoveCart")
    r = _movecart_direct(robot, phase0, tool, user, vel_list, reconnect=reconnect, label="phase0")
    if int(r) != 0:
        return {"ok": False, "msg": f"Phase0 이동 실패 err={r}", "phase0": phase0}

    # 2) phase0 도착 후 다시 joint 읽어서 target IK 확인 + 이동
    (ep2, _pose2), (ej2, joint_at_phase0) = rs.read_pose_joint(robot, reconnect=reconnect)
    if ep2 != 0 or ej2 != 0 or joint_at_phase0 is None:
        return {"ok": False, "msg": f"phase0 이후 상태 읽기 실패 err_p={ep2}, err_j={ej2}"}

    if not _has_solution(robot, target, joint_at_phase0, reconnect=reconnect):
        return {"ok": False, "msg": "Target IK 불가(하강 경로 막힘)", "target": target}

    print("[SMOOTH] 2) Target MoveCart (down)")
    r = _movecart_direct(robot, target, tool, user, vel_list, reconnect=reconnect, label="target")
    if int(r) != 0:
        return {"ok": False, "msg": f"Target 이동 실패 err={r}", "target": target}

    # 3) 그리퍼 닫기
    if auto_grip_close:
        print("[SMOOTH] 3) Gripper close")
        gc.gripper_close(robot, reconnect=reconnect, state=state)

    # 4) 완료 상태
    (e_end_p, pose_end), (e_end_j, joint_end) = rs.read_pose_joint(robot, reconnect=reconnect)
    if e_end_p == 0 and pose_end is not None:
        print("[SMOOTH] done pose:", rs.fmt_pose6(pose_end))

    return {
        "ok": True,
        "msg": "Smooth auto 완료",
        "phase0": phase0,
        "target": target,
        "pose_end": pose_end if e_end_p == 0 else None,
        "joint_end": joint_end if e_end_j == 0 else None,
    }


# ============================================================
# ✅ main 6번: 캐시 기반 1줄 호출용 (한 번에)
# - 4번(target_pose 캐시)
# - 5번(ik_check 캐시: phase0)
# ============================================================
def cmd6(robot, reconnect=None) -> Dict[str, Any]:
    """
    main.py:
      elif cmd == "6":
          sa.cmd6(cf.get_robot(), reconnect=_reconnect)
    """
    if robot is None:
        print("[6] Robot not connected. (0번 먼저)")
        return {"ok": False, "msg": "robot is None"}

    from . import target_pose as tp
    from . import ik_check as ik

    target = tp.get_last_target_pose6() if hasattr(tp, "get_last_target_pose6") else None
    phase0 = ik.get_last_phase0_pose6() if hasattr(ik, "get_last_phase0_pose6") else None

    if target is None:
        print("[6] last target 없음 (4번 먼저)")
        return {"ok": False, "msg": "last target is None"}

    if phase0 is None:
        print("[6] last phase0 없음 (5번에서 phase0 OK 먼저)")
        return {"ok": False, "msg": "last phase0 is None"}

    tool = int(getattr(rc, "TOOL_ID", 0))
    user = int(getattr(rc, "USER_ID", 0))
    auto_grip_close = bool(getattr(rc, "AUTO_GRIP_CLOSE", True))

    out = cmd_smooth_auto(
        robot,
        reconnect=reconnect,
        target_pose6=target,
        phase0_pose6=phase0,
        state=_STATE,
        tool=tool,
        user=user,
        auto_grip_close=auto_grip_close,
        vel_list=getattr(rc, "MOVE_CART_VEL_LIST", [20, 10, 5]),
    )

    if not out.get("ok"):
        print("[6] FAIL:", out.get("msg"))
    return out


# ============================================================
# ✅ main 7번: 2-step 토글
#  - 1회: phase0까지만 이동
#  - 2회: target으로 내려가고 그리퍼 닫기
# ============================================================
def cmd7(robot, reconnect=None) -> Dict[str, Any]:
    """
    main.py:
      elif cmd == "7":
          out = sa.cmd7(cf.get_robot(), reconnect=_reconnect)
    """
    if robot is None:
        print("[7] Robot not connected. (0번 먼저)")
        cmd7_reset()
        return {"ok": False, "msg": "robot is None"}

    from . import target_pose as tp
    from . import ik_check as ik

    target = tp.get_last_target_pose6() if hasattr(tp, "get_last_target_pose6") else None
    phase0 = ik.get_last_phase0_pose6() if hasattr(ik, "get_last_phase0_pose6") else None

    if target is None:
        print("[7] last target 없음 (4번 먼저)")
        cmd7_reset()
        return {"ok": False, "msg": "last target is None"}

    if phase0 is None:
        print("[7] last phase0 없음 (5번에서 phase0 OK 먼저)")
        cmd7_reset()
        return {"ok": False, "msg": "last phase0 is None"}

    tool = int(getattr(rc, "TOOL_ID", 0))
    user = int(getattr(rc, "USER_ID", 0))
    vel_list = list(getattr(rc, "MOVE_CART_VEL_LIST", [20, 10, 5]))
    auto_grip_close = bool(getattr(rc, "AUTO_GRIP_CLOSE", True))

    target = _ensure_pose6(target)
    phase0 = _ensure_pose6(phase0)

    # 이미 armed인데 캐시가 바뀌었으면(새 계산) 안전하게 리셋 후 1단계부터
    armed = bool(_CMD7_CTX.get("armed", False))
    if armed and (not _same_pose6(_CMD7_CTX.get("target"), target) or not _same_pose6(_CMD7_CTX.get("phase0"), phase0)):
        cmd7_reset()
        armed = False

    # ------------------------------------------------------------
    # 1회차: phase0까지만 이동하고 armed=True
    # ------------------------------------------------------------
    if not armed:
        (e1, _pose), (e2, cur_joint6) = rs.read_pose_joint(robot, reconnect=reconnect)
        if e1 != 0 or e2 != 0 or cur_joint6 is None:
            return {"ok": False, "msg": f"상태 읽기 실패 err_p={e1}, err_j={e2}"}

        if not _has_solution(robot, phase0, cur_joint6, reconnect=reconnect):
            return {"ok": False, "msg": "Phase0 IK 불가(이동 불가)", "phase0": phase0}

        print("\n[CMD7] 1) Phase0 MoveCart (ONLY)")
        r = _movecart_direct(robot, phase0, tool, user, vel_list, reconnect=reconnect, label="phase0")
        if int(r) != 0:
            return {"ok": False, "msg": f"Phase0 이동 실패 err={r}", "phase0": phase0}

        _CMD7_CTX["armed"] = True
        _CMD7_CTX["target"] = target
        _CMD7_CTX["phase0"] = phase0
        _CMD7_CTX["t"] = time.time()

        print("[CMD7] phase0 도착. 7번 한 번 더 누르면 내려가서 집습니다.")
        return {"ok": True, "msg": "phase0 reached (press 7 again to descend+grip)", "armed": True}

    # ------------------------------------------------------------
    # 2회차: target으로 내려가고 그리퍼 닫기
    # ------------------------------------------------------------
    (e1, _pose2), (e2, joint_at_phase0) = rs.read_pose_joint(robot, reconnect=reconnect)
    if e1 != 0 or e2 != 0 or joint_at_phase0 is None:
        return {"ok": False, "msg": f"phase0 이후 상태 읽기 실패 err_p={e1}, err_j={e2}"}

    if not _has_solution(robot, target, joint_at_phase0, reconnect=reconnect):
        cmd7_reset()
        return {"ok": False, "msg": "Target IK 불가(하강 경로 막힘)", "target": target, "armed": False}

    print("\n[CMD7] 2) Target MoveCart (DOWN)")
    r = _movecart_direct(robot, target, tool, user, vel_list, reconnect=reconnect, label="target")
    if int(r) != 0:
        cmd7_reset()
        return {"ok": False, "msg": f"Target 이동 실패 err={r}", "target": target, "armed": False}

    if auto_grip_close:
        print("[CMD7] 3) Gripper close")
        gc.gripper_close(robot, reconnect=reconnect, state=gc.get_state())

    (e_end_p, pose_end), _ = rs.read_pose_joint(robot, reconnect=reconnect)
    if e_end_p == 0 and pose_end is not None:
        print("[CMD7] done pose:", rs.fmt_pose6(pose_end))

    cmd7_reset()
    return {"ok": True, "msg": "descend+grip done", "armed": False, "pose_end": pose_end if e_end_p == 0 else None}


# ============================================================
# Optional: 상태/리셋 API
# ============================================================
def get_state() -> Dict[str, Any]:
    return gc.get_state()

def reset_state() -> None:
    gc.reset_state()
    cmd7_reset()
