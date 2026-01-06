import time
from robot import robot_config as rc
from robot import robot_state as rs

def _safe_call(fn, *args, reconnect=None, **kwargs):
    """RPC 호출 재시도 및 연결 관리 래퍼"""
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
    """IK(역기구학) 해 존재 여부 확인"""
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
    """MoveCart 실행 (속도 fallback 포함)"""
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
    """그리퍼 닫기 수행"""
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
    ✅ 수정된 자동 모드 (J6 별도 회전 제거)
    동작: Phase0(XY + Z_hold + RZ회전 동시 수행) -> Z-down(수직하강) -> Gripper Close
    """
    if last_target_pose6 is None:
        return {"ok": False, "msg": "target_pose 없음(3/4번 보정 먼저 수행 필요)"}

    # 보정된 최종 타겟 데이터 (이미 RZ 각도 포함됨)
    target = [float(x) for x in last_target_pose6[:6]]
    z_hold = float(target[2]) + float(rc.Z_HOLD_OFFSET_MM)

    # 0) 현재 상태 읽기
    (e1, cur_pose6), (e2, cur_joint6) = rs.read_pose_joint(robot, reconnect=reconnect)
    if e1 != 0 or e2 != 0:
        return {"ok": False, "msg": f"상태 읽기 실패 err_p={e1}, err_j={e2}"}

    # 1) Phase0 이동 (진입 높이까지 XY 이동 + RZ 각도 정렬 동시 수행)
    # build_target_pose에서 target_rz를 이미 계산했으므로 MoveCart 시 자동으로 회전함
    phase0 = target[:]
    phase0[2] = z_hold

    if not _has_solution(robot, phase0, cur_joint6, reconnect=reconnect):
        return {"ok": False, "msg": "Phase0 위치로의 IK 해가 없습니다.", "phase0": phase0}

    print("\n[CMD9] 1) Phase0 MoveCart (XY + Z_hold + RZ 정렬)")
    r = _movecart_direct(robot, phase0, tool, user, rc.MOVE_CART_VEL_LIST, reconnect=reconnect, label="phase0")
    if r != 0:
        return {"ok": False, "msg": f"Phase0 이동 실패 err={r}", "phase0": phase0}

    # 2) Z-down 이동 (수직 하강)
    # Phase0에서 각도가 이미 맞춰졌으므로 현재 pose에서 Z만 목표치로 변경
    (e1, pose_now), (e2, joint_now) = rs.read_pose_joint(robot, reconnect=reconnect)
    if e1 != 0 or e2 != 0:
        return {"ok": False, "msg": "하강 전 상태 읽기 실패"}

    down_pose = pose_now[:]
    down_pose[2] = float(target[2])

    if not _has_solution(robot, down_pose, joint_now, reconnect=reconnect):
        return {"ok": False, "msg": "Z-down 위치로의 IK 해가 없습니다.", "zdown": down_pose}

    print("[CMD9] 2) Z-down MoveCart (수직 하강)")
    r = _movecart_direct(robot, down_pose, tool, user, rc.MOVE_CART_VEL_LIST, reconnect=reconnect, label="zdown")
    if r != 0:
        return {"ok": False, "msg": f"Z-down 이동 실패 err={r}", "zdown": down_pose}

    # 3) 그리퍼 닫기
    if auto_grip_close:
        print("[CMD9] 3) Gripper close")
        _gripper_close_min(robot, reconnect=reconnect, state=state)

    # 최종 상태 확인 및 완료
    (e1, pose_end), (e2, joint_end) = rs.read_pose_joint(robot, reconnect=reconnect)
    if e1 == 0 and e2 == 0:
        print("[CMD9] 최종 위치 :", rs.fmt_pose6(pose_end))
    
    return {"ok": True, "msg": "Smooth auto 완료 (J6 회전 포함)", "pose_end": pose_end if e1 == 0 else None}