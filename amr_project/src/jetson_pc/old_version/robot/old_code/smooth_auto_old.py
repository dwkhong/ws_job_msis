# robot/smooth_auto.py
import time
from robot import robot_config as rc
from robot import robot_state as rs
from robot import j6_rotate as j6r


def _safe_call(fn, *args, reconnect=None, **kwargs):
    last_e = None
    for k in range(int(rc.RPC_RETRY) + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_e = e
            msg = str(e).lower()
            if ("timeout" in msg) or ("timed out" in msg) or ("实时数据失败" in str(e)):
                if reconnect is not None:
                    try:
                        reconnect()
                    except Exception:
                        pass
            if k < int(rc.RPC_RETRY):
                time.sleep(float(rc.RPC_RETRY_SLEEP_SEC))
                continue
            raise last_e


def _has_solution(robot, pose6, cur_joint6, reconnect=None) -> bool:
    pose6 = [float(x) for x in pose6[:6]]
    cur_joint6 = [float(x) for x in cur_joint6[:6]]
    err, ok = _safe_call(
        robot.GetInverseKinHasSolution,
        int(rc.IK_REF),
        pose6,
        cur_joint6,
        reconnect=reconnect,
    )
    return (err == 0) and bool(ok)


def _movecart_direct(robot, pose6, tool, user, vel_list, reconnect=None, label=""):
    pose6 = [float(x) for x in pose6[:6]]
    acc = 0.0
    ovl = 100.0

    for vv in list(vel_list):
        rtn = _safe_call(
            robot.MoveCart,
            pose6, tool, user, float(vv), acc, ovl, -1.0, -1,
            reconnect=reconnect,
        )
        if rtn == 0:
            if label:
                print(f"[MoveCart-OK] {label} vel={vv}")
            return 0
        if rtn == 112:
            if label:
                print(f"[MoveCart-112] {label} vel={vv} -> try next")
            continue
        if label:
            print(f"[MoveCart-FAIL] {label} vel={vv} err={rtn} -> try next")
    return 112


def _gripper_close_min(robot, reconnect=None, state=None):
    """
    gc 모듈 내부 함수 이름이 확실치 않아서,
    rc 상수 기준으로 여기서 최소 닫기만 수행.
    """
    try:
        if state is not None and not state.get("gripper_activated", False):
            err = _safe_call(robot.ActGripper, rc.GRIPPER_INDEX, 1, reconnect=reconnect)
            if err == 0:
                state["gripper_activated"] = True
            time.sleep(0.2)

        err = _safe_call(
            robot.MoveGripper,
            rc.GRIPPER_INDEX,
            int(rc.GRIP_CLOSE_POS),
            int(rc.GRIPPER_SPEED),
            int(rc.GRIPPER_FORCE),
            int(rc.GRIPPER_MAX_TIME),
            int(rc.GRIPPER_BLOCK),
            0, 0, 0, 0,
            reconnect=reconnect,
        )
        if state is not None and err == 0:
            state["gripper_closed"] = True
        print(f"[GRIP] close retval={err}")
        time.sleep(0.2)
        return (err == 0)
    except Exception as e:
        print(f"[GRIP] close exception: {e}")
        return False


def cmd9_smooth_auto(
    robot,
    reconnect,
    last_target_pose6,
    last_measure,
    state,
    tool=0,
    user=0,
    auto_grip_close=True,
):
    """
    전제: 1~4번까지 끝나서 last_target_pose6는 '보정된 target_pose' 상태.
    동작: phase0(=target z + hold) MoveCart -> J6 rotate -> Zdown MoveCart -> (옵션) gripper close
    """
    if last_target_pose6 is None:
        return {"ok": False, "msg": "target_pose 없음(3/4 먼저)"}

    target = [float(x) for x in last_target_pose6[:6]]
    z_hold = float(target[2]) + float(rc.Z_HOLD_OFFSET_MM)

    # 0) 현재 상태
    (e1, cur_pose6), (e2, cur_joint6) = rs.read_pose_joint(robot, reconnect=reconnect)
    if e1 != 0 or e2 != 0:
        return {"ok": False, "msg": f"상태 읽기 실패 err_p={e1}, err_j={e2}"}

    # 1) Phase0 pose
    phase0 = target[:]
    phase0[2] = z_hold

    if not _has_solution(robot, phase0, cur_joint6, reconnect=reconnect):
        return {"ok": False, "msg": "phase0 IK 불가(4번 보정 다시 필요)", "phase0": phase0}

    print("\n[CMD9] 1) Phase0 MoveCart (XY + Zhold)")
    r = _movecart_direct(robot, phase0, tool, user, rc.MOVE_CART_VEL_LIST, reconnect=reconnect, label="phase0")
    if r != 0:
        return {"ok": False, "msg": f"phase0 MoveCart 실패 err={r}", "phase0": phase0}

    # 2) J6 rotate
    print("[CMD9] 2) J6 rotate (angle_deg)")
    if last_measure is not None:
        ok, delta, err = j6r.rotate_j6_from_measure(robot=robot, last_measure=last_measure, reconnect=reconnect)
        if not ok:
            return {"ok": False, "msg": f"J6 rotate 실패 err={err}"}
        print(f"[CMD9] J6 rotate done delta={delta:+.3f} deg")
    else:
        print("[CMD9] last_measure 없음 -> 회전 스킵")

    # 3) Zdown (현재 pose 기준으로 Z만 target로)
    (e1, pose_after_rot), (e2, joint_after_rot) = rs.read_pose_joint(robot, reconnect=reconnect)
    if e1 != 0 or e2 != 0:
        return {"ok": False, "msg": f"회전 후 상태 읽기 실패 err_p={e1}, err_j={e2}"}

    down_pose = pose_after_rot[:]
    down_pose[2] = float(target[2])

    if not _has_solution(robot, down_pose, joint_after_rot, reconnect=reconnect):
        return {"ok": False, "msg": "zdown IK 불가(회전 영향/자세 확인)", "zdown": down_pose}

    print("[CMD9] 3) Zdown MoveCart (Z only)")
    r = _movecart_direct(robot, down_pose, tool, user, rc.MOVE_CART_VEL_LIST, reconnect=reconnect, label="zdown")
    if r != 0:
        return {"ok": False, "msg": f"zdown MoveCart 실패 err={r}", "zdown": down_pose}

    # 4) Gripper close
    if auto_grip_close:
        print("[CMD9] 4) Gripper close")
        _gripper_close_min(robot, reconnect=reconnect, state=state)

    # done state
    (e1, pose_end), (e2, joint_end) = rs.read_pose_joint(robot, reconnect=reconnect)
    if e1 == 0 and e2 == 0:
        print("[CMD9] done pose :", rs.fmt_pose6(pose_end))
        print("[CMD9] done joint:", rs.fmt_joint(joint_end))

    return {"ok": True, "msg": "Smooth auto 완료", "pose_end": pose_end if e1 == 0 else None}
