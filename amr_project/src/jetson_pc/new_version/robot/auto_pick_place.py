# robot/auto_pick_place.py
from __future__ import annotations

import time
from typing import Any, Dict, Sequence, Union, List

from . import robot_state as rs
from . import target_pose as tp
from . import ik_check as ik
from . import smooth_auto as sa
from . import return_home as rh
from . import gripper_control as gc
from . import robot_config as rc

from vision import measure_box_2 as mb

Pose6 = Union[List[float], Sequence[float]]


# ============================================================
# Helpers
# ============================================================
def _need_camera_running() -> bool:
    if hasattr(mb, "is_running"):
        try:
            return bool(mb.is_running())
        except Exception:
            return True
    return True


def _safe_call(fn, *args, reconnect=None, **kwargs):
    """
    최소 safe_call: 예외면 reconnect 1회 시도 후 재호출
    """
    try:
        return fn(*args, **kwargs)
    except Exception:
        if reconnect:
            try:
                reconnect()
            except Exception:
                pass
        return fn(*args, **kwargs)


def _set_vision_allow_restart(v: bool):
    """
    ✅ Auto 안정화용:
    - 비전 모듈이 지원하면(set_allow_restart) restart 허용/금지 토글
    - 지원 안 하면 조용히 무시
    """
    if hasattr(mb, "set_allow_restart"):
        try:
            mb.set_allow_restart(bool(v))
        except Exception:
            pass


def _get_home_joint6(robot, reconnect=None):
    """
    home_joint6 얻기:
    1) rs.get_initial_joint6() 있으면 그걸 사용
    2) 없으면 rs.get_last_joint6() (2번 눌렀을 때 홈이었다고 가정)
    3) 그래도 없으면 현재 joint 읽어서 사용 (최후수단)
    """
    if hasattr(rs, "get_initial_joint6"):
        hj = rs.get_initial_joint6()
        if hj is not None:
            return hj

    if hasattr(rs, "get_last_joint6"):
        hj = rs.get_last_joint6()
        if hj is not None:
            return hj

    # 최후수단: 현재 joint 읽기
    try:
        (e1, _p), (e2, j) = rs.read_pose_joint(robot, reconnect=reconnect)
        if e2 == 0 and j is not None:
            return j
    except Exception:
        pass

    return None


# ============================================================
# PLACE (inline version of stack_cycle_11)
# ============================================================
def _place_one_stack(robot, reconnect, state: Dict[str, Any], home_joint6, tool=0, user=0) -> Dict[str, Any]:
    """
    A(MoveJ) -> DROP(MoveCart) -> Gripper OPEN -> A(MoveJ) -> HOME(MoveJ) -> counter++
    """
    if home_joint6 is None:
        return {"ok": False, "msg": "home_joint6 없음 (2번으로 홈 위치 저장 필요)"}

    cnt = int(state.get("stack_counter", 0))

    drop = list(getattr(rc, "WP11_DROP_BASE_POSE"))
    drop[2] = float(drop[2]) + float(getattr(rc, "STACK_Z_STEP_MM", 0.0)) * cnt

    zmax = getattr(rc, "STACK_Z_MAX_MM", None)
    if zmax is not None and float(drop[2]) > float(zmax):
        return {"ok": False, "msg": f"DROP Z too high: {drop[2]:.1f} > {float(zmax):.1f}", "drop": drop}

    print(f"\n[PLACE] counter={cnt} dropZ={drop[2]:.1f}")

    # 1) A
    r = _safe_call(
        robot.MoveJ,
        joint_pos=getattr(rc, "WP11_A_JOINT"),
        tool=int(tool),
        user=int(user),
        vel=float(getattr(rc, "MOVEJ_VEL_WP11", 30.0)),
        blendT=float(getattr(rc, "MOVEJ_BLENDT_WP11", -1.0)),
        reconnect=reconnect,
    )
    if int(r) != 0:
        return {"ok": False, "msg": f"MoveJ(A) err={r}"}

    # 2) DROP (MoveCart)
    v0 = float(getattr(rc, "MOVE_CART_VEL_DEFAULT", 20.0))
    r = _safe_call(robot.MoveCart, drop, int(tool), int(user), v0, 0.0, 100.0, -1.0, -1, reconnect=reconnect)
    if int(r) != 0:
        # 112 등 실패 시 fallback 속도 재시도
        v_fbs = list(getattr(rc, "MOVE_CART_VEL_FALLBACKS", [10.0, 5.0]))
        for vv in v_fbs:
            r = _safe_call(
                robot.MoveCart, drop, int(tool), int(user),
                float(vv), 0.0, 100.0, -1.0, -1,
                reconnect=reconnect
            )
            if int(r) == 0:
                break

    if int(r) != 0:
        return {"ok": False, "msg": f"MoveCart(DROP) err={r}", "drop": drop}

    # 3) OPEN (놓기)
    try:
        gc.gripper_open(robot=robot, reconnect=reconnect, state=state)
    except Exception as e:
        return {"ok": False, "msg": f"gripper_open failed: {e}", "drop": drop}

    # 4) A back
    r = _safe_call(
        robot.MoveJ,
        joint_pos=getattr(rc, "WP11_A_JOINT"),
        tool=int(tool),
        user=int(user),
        vel=float(getattr(rc, "MOVEJ_VEL_WP11", 30.0)),
        blendT=float(getattr(rc, "MOVEJ_BLENDT_WP11", -1.0)),
        reconnect=reconnect,
    )
    if int(r) != 0:
        return {"ok": False, "msg": f"MoveJ(A back) err={r}"}

    # 5) HOME
    r = _safe_call(
        robot.MoveJ,
        joint_pos=home_joint6,
        tool=int(tool),
        user=int(user),
        vel=float(getattr(rc, "MOVEJ_VEL_RETURN", 30.0)),
        blendT=float(getattr(rc, "MOVEJ_BLENDT_RETURN", -1.0)),
        reconnect=reconnect,
    )
    if int(r) != 0:
        return {"ok": False, "msg": f"MoveJ(HOME) err={r}"}

    state["stack_counter"] = cnt + 1
    return {"ok": True, "msg": f"PLACE done. counter->{state['stack_counter']}", "drop": drop}


# ============================================================
# CMD12: Auto Pick & Place loop
# ============================================================
def cmd12(robot, reconnect=None) -> Dict[str, Any]:
    """
    ✅ 12번:
      N 입력받고
      HOME(정렬) ->
      [3] measure_avg ->
      [4] build target ->
      [5] IK check ->
      [6] smooth pick ->
      HOME only(그리퍼 유지) ->
      PLACE(A->DROP->OPEN->A->HOME, counter++)
      반복

    ✅ 안정화:
      - Step3(측정) 중에는 vision restart 허용
      - Step4~8(로봇 동작) 중에는 vision restart 금지 (크래시 방지)
      - 종료/에러/리턴 시에는 반드시 restart 허용으로 복구
    """
    if robot is None:
        print("[12] Robot not connected. (0번 먼저)")
        return {"ok": False, "msg": "robot is None"}

    if not _need_camera_running():
        print("[12] 카메라가 꺼져있습니다. (1번으로 ON)")
        return {"ok": False, "msg": "camera not running"}

    raw = input("박스 몇 개 옮길까요? (예: 4, b=back) > ").strip().lower()
    if raw in ("b", "back", "q", "quit"):
        return {"ok": True, "msg": "cancel"}

    try:
        n = int(raw)
        if n <= 0:
            raise ValueError()
    except Exception:
        print("[12] 숫자 입력이 아닙니다.")
        return {"ok": False, "msg": "invalid count"}

    state = gc.get_state()
    state.setdefault("stack_counter", 0)

    tool = int(getattr(rc, "TOOL_ID", 0))
    user = int(getattr(rc, "USER_ID", 0))

    home_joint6 = _get_home_joint6(robot, reconnect=reconnect)
    if home_joint6 is None:
        print("[12] home_joint6를 못 찾았음. 2번(홈 저장)부터 하세요.")
        return {"ok": False, "msg": "home_joint6 missing"}

    print(f"\n[12] Auto Pick&Place start: {n} cycles (stack_counter={state.get('stack_counter', 0)})")

    # ✅ 어떤 이유로든 cmd12가 끝나면 restart 허용으로 복구
    _set_vision_allow_restart(True)

    try:
        for i in range(n):
            print("\n" + "=" * 60)
            print(f"[12] Cycle {i+1}/{n}")
            print("=" * 60)

            # 0) 시작 HOME 정렬(그리퍼 유지)
            out_home0 = rh.cmd_home_only(robot, reconnect=reconnect)
            if not out_home0.get("ok", False):
                print("[12] HOME(prepare) FAIL:", out_home0.get("msg", ""))
                return {"ok": False, "msg": "home prepare fail", "cycle": i+1}

            # (권장) HOME 후 pose/joint 캐시 갱신
            try:
                rs.read_pose_joint(robot, reconnect=reconnect)
            except Exception:
                pass

            # 3) Measure (cache)  -> ✅ 측정 중에는 restart 허용
            print("[12] Step3: measure_avg")
            _set_vision_allow_restart(True)
            meas = mb.cmd_measure_avg()
            time.sleep(0.05)

            if meas is None:
                print("[12] measure_avg 실패")
                return {"ok": False, "msg": "measure fail", "cycle": i+1}

            # ✅ 이제부터 로봇이 움직일 거라 restart 금지
            _set_vision_allow_restart(False)

            # 4) Build target from last
            print("[12] Step4: build target from last")
            tp.cmd_build_target_from_last(robot, reconnect=reconnect, use_last_pose=True)

            if hasattr(tp, "get_last_target_pose6") and tp.get_last_target_pose6() is None:
                print("[12] target 생성 실패(캐시 없음)")
                return {"ok": False, "msg": "target cache missing", "cycle": i+1}

            # 5) IK check (cache: phase0)
            print("[12] Step5: IK check from last")
            ik.cmd_check_target_from_last(robot, reconnect=reconnect)

            if hasattr(ik, "get_last_phase0_pose6") and ik.get_last_phase0_pose6() is None:
                print("[12] IK 체크 실패(phase0 캐시 없음)")
                return {"ok": False, "msg": "phase0 cache missing", "cycle": i+1}

            # 6) Smooth pick
            print("[12] Step6: smooth pick (cmd6)")
            out_pick = sa.cmd6(robot, reconnect=reconnect, state=state)
            if not out_pick.get("ok", False):
                print("[12] PICK FAIL:", out_pick.get("msg", ""))
                return {"ok": False, "msg": "pick fail", "cycle": i+1, "detail": out_pick}

            # 7) HOME only (carry)
            print("[12] Step7: HOME only (carry box)")
            out_home1 = rh.cmd_home_only(robot, reconnect=reconnect)
            if not out_home1.get("ok", False):
                print("[12] HOME(carry) FAIL:", out_home1.get("msg", ""))
                return {"ok": False, "msg": "home carry fail", "cycle": i+1}

            # 8) PLACE inline
            print("[12] Step8: PLACE (A->DROP->OPEN->A->HOME)")
            out_place = _place_one_stack(
                robot=robot,
                reconnect=reconnect,
                state=state,
                home_joint6=home_joint6,
                tool=tool,
                user=user,
            )
            if not out_place.get("ok", False):
                print("[12] PLACE FAIL:", out_place.get("msg", ""))
                return {"ok": False, "msg": "place fail", "cycle": i+1, "detail": out_place}

            print(f"[12] ✅ Cycle {i+1} done. stack_counter={state.get('stack_counter')}")

            # ✅ 다음 사이클 측정 전에 restart 허용으로 다시 풀어둠(안전)
            _set_vision_allow_restart(True)

        print("\n[12] 🎉 All cycles done.")
        return {"ok": True, "msg": "done", "count": n, "stack_counter": state.get("stack_counter", 0)}

    finally:
        # ✅ 어떤 종료 경로든 vision restart 허용 복구
        _set_vision_allow_restart(True)
