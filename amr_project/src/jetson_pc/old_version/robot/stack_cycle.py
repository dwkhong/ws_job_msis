# robot/stack_cycle_11.py
import time
from robot import robot_config as rc
from robot import robot_state as rs
from robot import gripper_control as gc  # 있으면 씀(없으면 아래 fallback)


def _safe_call(fn, *args, reconnect=None, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        if reconnect:
            reconnect()
        return fn(*args, **kwargs)


def _gripper_open_fallback(robot, reconnect=None, state=None):
    # gc.gripper_open 없을 때 최소 동작
    if state is not None and not state.get("gripper_activated", False):
        _safe_call(robot.ActGripper, rc.GRIPPER_INDEX, 1, reconnect=reconnect)
        state["gripper_activated"] = True
        time.sleep(0.2)

    _safe_call(
        robot.MoveGripper,
        rc.GRIPPER_INDEX,
        int(rc.GRIP_OPEN_POS),
        int(rc.GRIPPER_SPEED),
        int(rc.GRIPPER_FORCE),
        int(rc.GRIPPER_MAX_TIME),
        int(rc.GRIPPER_BLOCK),
        0, 0, 0, 0,
        reconnect=reconnect
    )
    if state is not None:
        state["gripper_closed"] = False
    time.sleep(0.2)


def cmd11_stack_cycle(robot, reconnect, state, home_joint6, tool=0, user=0):
    """
    A(MoveJ) -> DROP(MoveCart) -> Gripper OPEN -> A(MoveJ) -> HOME(MoveJ) -> counter++
    """
    if home_joint6 is None:
        return {"ok": False, "msg": "home_joint6 없음(1번으로 initial_joint6 저장 필요)"}

    cnt = int(state.get("stack_counter", 0))
    drop = list(rc.WP11_DROP_BASE_POSE)
    drop[2] = float(drop[2]) + float(rc.STACK_Z_STEP_MM) * cnt

    if rc.STACK_Z_MAX_MM is not None and float(drop[2]) > float(rc.STACK_Z_MAX_MM):
        return {"ok": False, "msg": f"DROP Z too high: {drop[2]:.1f} > {rc.STACK_Z_MAX_MM}"}

    print(f"\n[CMD11] counter={cnt} dropZ={drop[2]:.1f}")

    # 1) A
    r = _safe_call(robot.MoveJ, joint_pos=rc.WP11_A_JOINT, tool=tool, user=user,
                   vel=rc.MOVEJ_VEL_WP11, blendT=rc.MOVEJ_BLENDT_WP11, reconnect=reconnect)
    if r != 0:
        return {"ok": False, "msg": f"MoveJ(A) err={r}"}

    # 2) DROP (IK 체크는 생략/간소화: 필요하면 여기만 추가하면 됨)
    r = _safe_call(robot.MoveCart, drop, tool, user,
                   float(rc.MOVE_CART_VEL_DEFAULT), 0.0, 100.0, -1.0, -1, reconnect=reconnect)
    if r != 0:
        # 112면 fallback 속도로 재시도
        for vv in rc.MOVE_CART_VEL_FALLBACKS:
            r = _safe_call(robot.MoveCart, drop, tool, user, float(vv), 0.0, 100.0, -1.0, -1, reconnect=reconnect)
            if r == 0:
                break
    if r != 0:
        return {"ok": False, "msg": f"MoveCart(DROP) err={r}", "drop": drop}

    # 3) OPEN
    if hasattr(gc, "gripper_open"):
        try:
            gc.gripper_open(robot=robot, reconnect=reconnect, state=state)
        except Exception:
            _gripper_open_fallback(robot, reconnect=reconnect, state=state)
    else:
        _gripper_open_fallback(robot, reconnect=reconnect, state=state)

    # 4) A back
    r = _safe_call(robot.MoveJ, joint_pos=rc.WP11_A_JOINT, tool=tool, user=user,
                   vel=rc.MOVEJ_VEL_WP11, blendT=rc.MOVEJ_BLENDT_WP11, reconnect=reconnect)
    if r != 0:
        return {"ok": False, "msg": f"MoveJ(A back) err={r}"}

    # 5) HOME
    r = _safe_call(robot.MoveJ, joint_pos=home_joint6, tool=tool, user=user,
                   vel=rc.MOVEJ_VEL_RETURN, blendT=rc.MOVEJ_BLENDT_RETURN, reconnect=reconnect)
    if r != 0:
        return {"ok": False, "msg": f"MoveJ(HOME) err={r}"}

    state["stack_counter"] = cnt + 1
    return {"ok": True, "msg": f"CMD11 done. counter->{state['stack_counter']}", "drop": drop}
