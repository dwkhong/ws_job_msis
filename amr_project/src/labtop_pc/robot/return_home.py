# robot/return_home.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from . import robot_state as rs
from . import robot_config as rc


def move_to_joint(
    robot,
    joint_target6: List[float],
    reconnect=None,
    label: str = "RETURN",
    vel: Optional[float] = None,
    blendT: Optional[float] = None,
) -> Tuple[bool, int]:
    """
    Fairino MoveJ wrapper
    return: (ok, errcode)
    """
    jt = rs.ensure_joint6(joint_target6)

    if vel is None:
        vel = float(rc.MOVEJ_VEL_RETURN)
    if blendT is None:
        blendT = float(rc.MOVEJ_BLENDT_RETURN)

    # 현재 joint 읽어서 delta 출력(선택)
    try:
        err_j, cur_joint = rs.safe_call(robot.GetActualJointPosDegree, flag=1, reconnect_cb=reconnect)
        if err_j == 0:
            cur6 = rs.ensure_joint6(cur_joint)
            dlt = [float(a) - float(b) for a, b in zip(jt[:6], cur6[:6])]
            print(f"[MOVEJ-{label}] current:", rs.fmt_joint(cur6))
            print(f"[MOVEJ-{label}] target :", rs.fmt_joint(jt))
            print("[MOVEJ-{label}] delta  :", "[" + ", ".join(f"{v:+.3f}" for v in dlt) + "]")
        else:
            print(f"[MOVEJ-{label}] (warn) cannot read current joint err={err_j}")
    except Exception as e:
        print(f"[MOVEJ-{label}] (warn) read current joint exception: {e}")

    # MoveJ 실행
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
    return (rtn == 0), int(rtn)


def cmd7_return_to_initial(
    robot,
    initial_joint6: Optional[List[float]],
    reconnect=None,
    reset_after: bool = True,
) -> Dict[str, object]:
    """
    ✅ CMD8: 프로그램 시작 초기 joint(initial_joint6)로 복귀
    out:
      ok(bool), msg(str), err(int), reset(bool)
    """
    out: Dict[str, object] = {"ok": False, "msg": "", "err": -1, "reset": False}

    if initial_joint6 is None:
        out["msg"] = "초기 joint(initial_joint6)가 저장되어 있지 않습니다."
        return out

    print("\n[ACTION] CMD8: 초기 위치로 복귀 (MoveJ -> initial_joint6)")
    ok, err = move_to_joint(robot, initial_joint6, reconnect=reconnect, label="INIT")
    out["ok"] = ok
    out["err"] = err

    if ok:
        out["msg"] = "초기 위치 복귀 완료"
        out["reset"] = bool(reset_after)
    else:
        out["msg"] = f"초기 위치 복귀 실패 errcode={err}"

    return out
