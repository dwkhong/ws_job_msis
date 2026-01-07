# robot/return_home.py (추가/정리용)

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

from . import robot_state as rs
from . import robot_config as rc
from . import gripper_control as gc


def move_to_joint(
    robot,
    joint_target6: List[float],
    reconnect=None,
    label: str = "RETURN",
    vel: Optional[float] = None,
    blendT: Optional[float] = None,
) -> Tuple[bool, int]:
    jt = rs.ensure_joint6(joint_target6)

    if vel is None:
        vel = float(getattr(rc, "MOVEJ_VEL_RETURN", 30.0))
    if blendT is None:
        blendT = float(getattr(rc, "MOVEJ_BLENDT_RETURN", -1.0))

    # 현재 joint 출력(선택)
    try:
        err_j, cur_joint = rs.safe_call(robot.GetActualJointPosDegree, flag=1, reconnect_cb=reconnect)
        if err_j == 0:
            cur6 = rs.ensure_joint6(cur_joint)
            dlt = [float(a) - float(b) for a, b in zip(jt[:6], cur6[:6])]
            print(f"[MOVEJ-{label}] current:", rs.fmt_joint(cur6))
            print(f"[MOVEJ-{label}] target :", rs.fmt_joint(jt))
            print(f"[MOVEJ-{label}] delta  :", "[" + ", ".join(f"{v:+.3f}" for v in dlt) + "]")
    except Exception:
        pass

    rtn = rs.safe_call(
        robot.MoveJ,
        joint_pos=jt,
        tool=int(getattr(rc, "TOOL_ID", 0)),
        user=int(getattr(rc, "USER_ID", 0)),
        vel=float(vel),
        blendT=float(blendT),
        reconnect_cb=reconnect
    )
    print(f"[RET] MoveJ({label}) errcode: {rtn}")
    return (int(rtn) == 0), int(rtn)


def cmd_return_to_initial(
    robot,
    initial_joint6: Optional[List[float]],
    reconnect=None,
    reset_after: bool = True,
) -> Dict[str, object]:
    out: Dict[str, object] = {"ok": False, "msg": "", "err": -1, "reset": False}

    if initial_joint6 is None:
        out["msg"] = "초기 joint(initial_joint6)가 저장되어 있지 않습니다."
        return out

    print("\n[ACTION] HOME only (MoveJ -> initial_joint6)")
    ok, err = move_to_joint(robot, initial_joint6, reconnect=reconnect, label="HOME")
    out["ok"] = ok
    out["err"] = err
    out["reset"] = bool(reset_after)
    out["msg"] = "HOME 복귀 완료" if ok else f"HOME 복귀 실패 errcode={err}"
    return out


def cmd8_return_home_and_open_gripper(
    robot,
    initial_joint6: Optional[List[float]],
    reconnect=None,
    reset_after: bool = True,
) -> Dict[str, object]:
    """
    ✅ main 8번용: HOME + GRIPPER OPEN
    """
    out = cmd_return_to_initial(robot, initial_joint6, reconnect=reconnect, reset_after=reset_after)
    if out.get("ok", False):
        print("[ACTION] HOME 도착 → Gripper OPEN")
        gc.gripper_open(robot, reconnect=reconnect, state=gc.get_state())
    return out


def cmd8(robot, reconnect=None) -> Dict[str, object]:
    """
    ✅ main 8번용 엔트리: 내부에서 initial_joint 없으면 캡처 시도 후 HOME+OPEN
    """
    if robot is None:
        return {"ok": False, "msg": "robot is None (0번으로 연결)", "err": -1, "reset": False}

    # initial_joint6 없으면 캡처 (robot_state에 있으면)
    if hasattr(rs, "get_initial_joint6") and hasattr(rs, "try_capture_initial_joint"):
        if rs.get_initial_joint6() is None:
            rs.try_capture_initial_joint(robot, reconnect=reconnect, verbose=False)
        initial = rs.get_initial_joint6()
    else:
        initial = None

    return cmd8_return_home_and_open_gripper(robot, initial, reconnect=reconnect, reset_after=True)


def cmd_home_only(robot, reconnect=None) -> Dict[str, object]:
    """
    ✅ 12번 루프용: HOME만(그리퍼 유지) 복귀
    """
    if robot is None:
        return {"ok": False, "msg": "robot is None", "err": -1}

    if hasattr(rs, "get_initial_joint6") and hasattr(rs, "try_capture_initial_joint"):
        if rs.get_initial_joint6() is None:
            rs.try_capture_initial_joint(robot, reconnect=reconnect, verbose=False)
        initial = rs.get_initial_joint6()
    else:
        initial = None

    return cmd_return_to_initial(robot, initial, reconnect=reconnect, reset_after=False)

