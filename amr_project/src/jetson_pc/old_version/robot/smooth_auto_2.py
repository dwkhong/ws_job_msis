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
    safe_phase0_pose,  # <--- [추가됨] 외부에서 계산한 안전 진입점
    last_measure,
    state,
    tool=0,
    user=0,
    auto_grip_close=True,
):
    """
    ✅ 수정된 자동 모드
    1. 인자로 받은 safe_phase0_pose로 이동 (계산된 안전 공중 자세)
    2. last_target_pose6로 직선 하강 (내려가면서 각도 자연 보정)
    """
    if last_target_pose6 is None or safe_phase0_pose is None:
        return {"ok": False, "msg": "좌표 데이터 부족(계산된 target/phase0 필요)"}

    # 최종 타겟 (바닥)
    target = [float(x) for x in last_target_pose6[:6]]
    
    # 0) 현재 상태 읽기
    (e1, cur_pose6), (e2, cur_joint6) = rs.read_pose_joint(robot, reconnect=reconnect)
    if e1 != 0 or e2 != 0:
        return {"ok": False, "msg": f"상태 읽기 실패 err_p={e1}, err_j={e2}"}

    # -----------------------------------------------------------
    # 1) Phase0 이동 (외부에서 계산된 안전한 공중 위치 )
    # -----------------------------------------------------------
    # 이미 IK가 검증된 safe_phase0_pose를 사용
    if not _has_solution(robot, safe_phase0_pose, cur_joint6, reconnect=reconnect):
         return {"ok": False, "msg": "Phase0 위치 IK 불가 (이동 불가)", "phase0": safe_phase0_pose}

    print("\n[CMD9] 1) Phase0 MoveCart (Safe Approach)")
    r = _movecart_direct(robot, safe_phase0_pose, tool, user, rc.MOVE_CART_VEL_LIST, reconnect=reconnect, label="phase0")
    if r != 0:
        return {"ok": False, "msg": f"Phase0 이동 실패 err={r}", "phase0": safe_phase0_pose}

    # -----------------------------------------------------------
    # 2) Z-down 이동 (Target으로 직접 이동) 
    # -----------------------------------------------------------
    # 기존처럼 Z만 내리는 게 아니라, Target 좌표로 직접 쏴줍니다.
    # 그래야 공중(Phase0)과 바닥(Target)의 각도가 다를 때(Safe Tilt 전략) 부드럽게 변하며 내려갑니다.
    
    # 하강 직전 관절 상태 읽기 (IK 확인용)
    _, current_joints_at_phase0 = rs.read_pose_joint(robot, reconnect=reconnect)
    
    if not _has_solution(robot, target, current_joints_at_phase0[1], reconnect=reconnect):
        return {"ok": False, "msg": "Target(Z-down) 위치 IK 불가 (하강 경로 막힘)", "target": target}

    print("[CMD9] 2) Z-down MoveCart (Target Approach)")
    r = _movecart_direct(robot, target, tool, user, rc.MOVE_CART_VEL_LIST, reconnect=reconnect, label="zdown")
    if r != 0:
        return {"ok": False, "msg": f"Z-down 이동 실패 err={r}", "target": target}

    # 3) 그리퍼 닫기
    if auto_grip_close:
        print("[CMD9] 3) Gripper close")
        _gripper_close_min(robot, reconnect=reconnect, state=state)

    # 최종 상태 확인 및 완료
    (e1, pose_end), (e2, joint_end) = rs.read_pose_joint(robot, reconnect=reconnect)
    if e1 == 0:
        print("[CMD9] 최종 위치 :", rs.fmt_pose6(pose_end))
    
    return {"ok": True, "msg": "Smooth auto 완료", "pose_end": pose_end if e1 == 0 else None}