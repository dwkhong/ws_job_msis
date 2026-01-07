# robot/gripper_control.py
from __future__ import annotations
import time
from typing import Dict, Any, Optional

# ✅ 패키지/단일 실행 모두 대비
try:
    from . import robot_config as rc
    from . import robot_state as rs
except Exception:
    import robot_config as rc
    import robot_state as rs


# ✅ 공용 상태(모든 모듈이 이걸 공유)
_STATE: Dict[str, Any] = {
    "gripper_activated": False,
    "gripper_closed": None,  # True/False/None(unknown)
}


def get_state() -> Dict[str, Any]:
    return _STATE


def reset_state() -> None:
    _STATE["gripper_activated"] = False
    _STATE["gripper_closed"] = None


def ensure_gripper_activated(robot, reconnect=None, state: Optional[Dict[str, Any]] = None) -> bool:
    """
    Gripper 활성화가 안 되어 있으면 ActGripper 호출.
    state=None이면 모듈 공용 _STATE 사용.
    """
    if state is None:
        state = _STATE

    if state.get("gripper_activated", False):
        return True

    print("[GRIP] Activating gripper...")
    try:
        err = rs.safe_call(
            robot.ActGripper,
            rc.GRIPPER_INDEX,
            1,
            retry=getattr(rc, "RPC_RETRY", 1),
            sleep_sec=getattr(rc, "RPC_RETRY_SLEEP_SEC", 0.25),
            reconnect_cb=reconnect
        )
    except Exception as e:
        print(f"[GRIP-FAIL] ActGripper exception: {e}")
        return False

    print("[GRIP] ActGripper:", err)
    if int(err) == 0:
        state["gripper_activated"] = True

    time.sleep(0.3)
    return int(err) == 0


def gripper_move(robot, pos: int, reconnect=None) -> int:
    """
    MoveGripper 호출 (pos만 입력받음)
    """
    try:
        err = rs.safe_call(
            robot.MoveGripper,
            rc.GRIPPER_INDEX,
            int(pos),
            int(rc.GRIPPER_SPEED),
            int(rc.GRIPPER_FORCE),
            int(rc.GRIPPER_MAX_TIME),
            int(rc.GRIPPER_BLOCK),
            0, 0, 0, 0,
            retry=getattr(rc, "RPC_RETRY", 1),
            sleep_sec=getattr(rc, "RPC_RETRY_SLEEP_SEC", 0.25),
            reconnect_cb=reconnect
        )
        return int(err)
    except Exception as e:
        print(f"[GRIP-FAIL] MoveGripper exception: {e}")
        return -999


def gripper_open(robot, reconnect=None, state: Optional[Dict[str, Any]] = None) -> bool:
    if state is None:
        state = _STATE

    if not ensure_gripper_activated(robot, reconnect=reconnect, state=state):
        return False

    print("[GRIP] Opening gripper...")
    err = gripper_move(robot, rc.GRIP_OPEN_POS, reconnect=reconnect)
    print("[GRIP] Open retval:", err)

    if err == 0:
        state["gripper_closed"] = False

    time.sleep(0.3)
    return err == 0


def gripper_close(robot, reconnect=None, state: Optional[Dict[str, Any]] = None) -> bool:
    if state is None:
        state = _STATE

    if not ensure_gripper_activated(robot, reconnect=reconnect, state=state):
        return False

    print("[GRIP] Closing gripper...")
    err = gripper_move(robot, rc.GRIP_CLOSE_POS, reconnect=reconnect)
    print("[GRIP] Close retval:", err)

    if err == 0:
        state["gripper_closed"] = True

    time.sleep(0.3)
    return err == 0


def gripper_toggle(robot, reconnect=None, state: Optional[Dict[str, Any]] = None) -> bool:
    if state is None:
        state = _STATE

    closed = state.get("gripper_closed", None)
    if closed is None:
        print("[GRIP] toggle: state unknown -> CLOSE first")
        return gripper_close(robot, reconnect=reconnect, state=state)

    return gripper_open(robot, reconnect=reconnect, state=state) if closed else gripper_close(robot, reconnect=reconnect, state=state)


def prompt_gripper_menu() -> str:
    print("\n---------------------------------------")
    print("Gripper Control")
    print("  o : Open")
    print("  c : Close")
    print("  t : Toggle")
    print("  b : Back")
    print("---------------------------------------")
    return input("gripper (o/c/t/b) > ").strip().lower()


def run_gripper_menu(robot, reconnect=None, state: Optional[Dict[str, Any]] = None):
    """
    main.py에서 cmd==10일 때 이 함수 하나만 호출하면 됨.
    state=None이면 공용 _STATE 사용.
    """
    if state is None:
        state = _STATE

    while True:
        gcmd = prompt_gripper_menu()
        if gcmd == "b":
            break
        elif gcmd == "o":
            gripper_open(robot, reconnect=reconnect, state=state)
        elif gcmd == "c":
            gripper_close(robot, reconnect=reconnect, state=state)
        elif gcmd == "t":
            gripper_toggle(robot, reconnect=reconnect, state=state)
        else:
            print("[WARN] invalid gripper cmd")


def cmd9(robot, reconnect=None) -> Dict[str, Any]:
    """
    ✅ main의 9번용: 그리퍼 메뉴만 실행 (내부 state 자동 사용)
    """
    if robot is None:
        print("[9] Robot not connected. (0번 먼저)")
        return {"ok": False, "msg": "robot is None"}

    run_gripper_menu(robot, reconnect=reconnect, state=get_state())
    return {"ok": True, "msg": "done"}
