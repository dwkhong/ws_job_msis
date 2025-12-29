# robot/gripper_control.py
import time

from robot import robot_config as rc
from robot import robot_state as rs


def ensure_gripper_activated(robot, reconnect=None, state=None) -> bool:
    """
    Gripper 활성화가 안 되어 있으면 ActGripper 호출.
    state dict를 넘겨주면 "gripper_activated" 플래그를 내부에서 관리함.
    """
    if state is not None and state.get("gripper_activated", False):
        return True

    print("[GRIP] Activating gripper...")
    try:
        err = rs.safe_call(
            robot.ActGripper,
            rc.GRIPPER_INDEX,
            1,
            retry=rc.RPC_RETRY,
            sleep_sec=rc.RPC_RETRY_SLEEP_SEC,
            reconnect_cb=reconnect
        )
    except Exception as e:
        print(f"[GRIP-FAIL] ActGripper exception: {e}")
        return False

    print("[GRIP] ActGripper:", err)
    if err == 0 and state is not None:
        state["gripper_activated"] = True

    time.sleep(0.3)
    return (err == 0)


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
            retry=rc.RPC_RETRY,
            sleep_sec=rc.RPC_RETRY_SLEEP_SEC,
            reconnect_cb=reconnect
        )
        return err
    except Exception as e:
        print(f"[GRIP-FAIL] MoveGripper exception: {e}")
        return -999


def gripper_open(robot, reconnect=None, state=None) -> bool:
    if not ensure_gripper_activated(robot, reconnect=reconnect, state=state):
        return False
    print("[GRIP] Opening gripper...")
    err = gripper_move(robot, rc.GRIP_OPEN_POS, reconnect=reconnect)
    print("[GRIP] Open retval:", err)
    if err == 0 and state is not None:
        state["gripper_closed"] = False
    time.sleep(0.3)
    return (err == 0)


def gripper_close(robot, reconnect=None, state=None) -> bool:
    if not ensure_gripper_activated(robot, reconnect=reconnect, state=state):
        return False
    print("[GRIP] Closing gripper...")
    err = gripper_move(robot, rc.GRIP_CLOSE_POS, reconnect=reconnect)
    print("[GRIP] Close retval:", err)
    if err == 0 and state is not None:
        state["gripper_closed"] = True
    time.sleep(0.3)
    return (err == 0)


def gripper_toggle(robot, reconnect=None, state=None) -> bool:
    closed = None if state is None else state.get("gripper_closed", None)
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


def run_gripper_menu(robot, reconnect=None, state=None):
    """
    main.py에서 cmd==10일 때 이 함수 하나만 호출하면 됨.
    """
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
