# robot/move_step.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from . import robot_state as rs
from . import robot_config as rc


def _has_solution(robot, pose6: List[float], cur_joint6: List[float], reconnect=None) -> bool:
    """
    IK solvable check
    Fairino: GetInverseKinHasSolution(ref=0, pose, cur_joint)
    """
    pose6 = rs.ensure_pose6(pose6)
    cur_joint6 = rs.ensure_joint6(cur_joint6)

    err, ok = rs.safe_call(
        robot.GetInverseKinHasSolution,
        0, pose6, cur_joint6,
        reconnect_cb=reconnect
    )
    if err != 0:
        return False
    return bool(ok)


def _joint_delta_str(j_new: Optional[List[float]], j_old: Optional[List[float]]) -> str:
    if j_new is None or j_old is None:
        return "(delta N/A)"
    d = [float(a) - float(b) for a, b in zip(j_new[:6], j_old[:6])]
    return "[" + ", ".join(f"{v:+.3f}" for v in d) + "]"


def _blend_pose_axis(cur_pose6: List[float],
                     target_pose6: List[float],
                     scale_xyz: Tuple[float, float, float],
                     ori_scale: float) -> List[float]:
    """
    axis-wise partial step
      - xyz: clamp towards target
      - rpy: linear blend with ori_scale
    """
    cur = rs.ensure_pose6(cur_pose6)
    tgt = rs.ensure_pose6(target_pose6)
    sx, sy, sz = [float(v) for v in scale_xyz]
    ori_scale = float(ori_scale)

    out = cur[:]

    # XYZ step with clamping
    for idx, s in zip([0, 1, 2], [sx, sy, sz]):
        delta = tgt[idx] - cur[idx]
        step = delta * s
        if delta >= 0:
            out[idx] = min(cur[idx] + step, tgt[idx])
        else:
            out[idx] = max(cur[idx] + step, tgt[idx])

    # RPY blend
    for idx in [3, 4, 5]:
        out[idx] = cur[idx] + (tgt[idx] - cur[idx]) * ori_scale

    return out


def _xy_reached(cur_pose6: List[float], target_pose6: List[float], tol_mm: float) -> bool:
    cur = rs.ensure_pose6(cur_pose6)
    tgt = rs.ensure_pose6(target_pose6)
    return (abs(cur[0] - tgt[0]) <= tol_mm) and (abs(cur[1] - tgt[1]) <= tol_mm)


def _z_reached(cur_pose6: List[float], z_target: float, tol_mm: float) -> bool:
    cur = rs.ensure_pose6(cur_pose6)
    return abs(cur[2] - float(z_target)) <= tol_mm


def _try_movecart(robot, reconnect, tool: int, user: int, pose6: List[float], vel_list: List[float]) -> int:
    """
    MoveCart 한번 시도(vel fallback 포함)
    return: 0 ok, 112 or other fail
    """
    pose6 = rs.ensure_pose6(pose6)

    for vv in vel_list:
        rtn = rs.safe_call(
            robot.MoveCart,
            pose6,
            int(tool),
            int(user),
            float(vv),
            float(rc.MOVE_CART_ACC),
            float(rc.MOVE_CART_OVL),
            float(rc.MOVE_CART_BLENDT),
            int(rc.MOVE_CART_EX),
            reconnect_cb=reconnect
        )
        if rtn == 0:
            return 0
        if rtn == 112:
            continue
        # 그 외도 다음 vel로 계속 시도(너 기존 정책과 동일)
        continue
    return 112


def do_one_movecart_step(
    robot,
    reconnect,
    tool: int,
    user: int,
    cur_pose6: List[float],
    cur_joint6: List[float],
    target_pose6: List[float],
    approach_phase: int,
    step_try_list: List[float],
    vel_list: List[float],
    step_scale_default: float,
) -> Tuple[bool, Optional[List[float]], Optional[List[float]], int, bool, Dict[str, Any]]:
    """
    ✅ CMD7: 버튼 1번 눌렀을 때 MoveCart "1-step"만 수행
    return:
      moved(bool),
      pose_after(or None),
      joint_after(or None),
      new_phase(int),
      done(bool),
      debug(dict)
    """
    dbg: Dict[str, Any] = {}
    target_pose6 = rs.ensure_pose6(target_pose6)
    cur_pose6 = rs.ensure_pose6(cur_pose6)
    cur_joint6 = rs.ensure_joint6(cur_joint6)

    z_hold = float(target_pose6[2]) + float(rc.Z_HOLD_OFFSET_MM)

    # Phase definition
    if int(approach_phase) == 0:
        # phase0: XY 맞추면서 Z는 z_hold
        phase_target_pose = target_pose6[:]
        phase_target_pose[2] = z_hold

        xy_scale_candidates = [1.0]     # XY 적극 이동
        ori_scale_candidates = [1.0]    # 자세도 target 따라감(기존 그대로)
    else:
        # phase1: Z만 내려감
        phase_target_pose = target_pose6[:]

        xy_scale_candidates = [0.0, 0.01, 0.02]  # XY 거의 고정
        ori_scale_candidates = [0.0]             # 자세 고정

    for st in step_try_list:
        st = float(st)

        for xy_s in xy_scale_candidates:
            xy_s = float(xy_s)

            if int(approach_phase) == 0:
                sx = min(1.0, st * float(rc.X_SCALE_MULT))
                sy = st
                sz = st
            else:
                sx = xy_s
                sy = xy_s
                sz = st

            for ori_s in ori_scale_candidates:
                ori_s = float(ori_s)

                step_pose = rs.ensure_pose6(
                    _blend_pose_axis(cur_pose6, phase_target_pose, (sx, sy, sz), ori_scale=ori_s)
                )

                # IK check
                if not _has_solution(robot, step_pose, cur_joint6, reconnect=reconnect):
                    continue

                # MoveCart try
                rtn = _try_movecart(robot, reconnect, tool, user, step_pose, vel_list)
                if rtn != 0:
                    continue

                # after state
                (e1, pose_after), (e2, joint_after) = rs.read_pose_joint(robot, reconnect=reconnect)
                if e1 != 0 or e2 != 0:
                    # 이동은 했는데 상태 못 읽음
                    dbg.update({"used_st": st, "sx": sx, "sy": sy, "sz": sz, "ori_s": ori_s})
                    return True, None, None, int(approach_phase), False, dbg

                new_phase = int(approach_phase)
                done = False

                if int(approach_phase) == 0:
                    xy_ok = _xy_reached(pose_after, target_pose6, tol_mm=float(rc.XY_TOL_MM))
                    z_ok = _z_reached(pose_after, z_hold, tol_mm=float(rc.Z_TOL_MM))
                    if xy_ok and z_ok:
                        new_phase = 1
                else:
                    z_done = _z_reached(pose_after, float(target_pose6[2]), tol_mm=float(rc.Z_TOL_MM))
                    if z_done:
                        done = True
                        new_phase = 0

                dbg.update({"used_st": st, "sx": sx, "sy": sy, "sz": sz, "ori_s": ori_s})
                return True, pose_after, joint_after, new_phase, done, dbg

    return False, None, None, int(approach_phase), False, dbg


def cmd7_run(
    robot,
    reconnect,
    last_target_pose6: List[float],
    approach_phase: int,
    reached_final: bool,
    step_scale: Optional[float] = None
) -> Dict[str, Any]:
    """
    main.py에서 쓰기 편하게 래핑한 CMD7 엔트리
    """
    out: Dict[str, Any] = {
        "ok": False,
        "moved": False,
        "done": False,
        "new_phase": int(approach_phase),
        "reached_final": bool(reached_final),
        "pose_after": None,
        "joint_after": None,
        "debug": {},
        "msg": "",
    }

    if reached_final:
        out["ok"] = True
        out["msg"] = "이미 최종(targetZ) 도달 상태"
        return out

    (e1, cur_pose6), (e2, cur_joint6) = rs.read_pose_joint(robot, reconnect=reconnect)
    if e1 != 0 or e2 != 0:
        out["msg"] = f"상태 읽기 실패 err_p={e1}, err_j={e2}"
        return out

    if step_scale is None:
        step_scale = float(rc.STEP_SCALE_DEFAULT)

    moved, pose_after, joint_after, new_phase, done, dbg = do_one_movecart_step(
        robot=robot,
        reconnect=reconnect,
        tool=int(rc.TOOL_ID),
        user=int(rc.USER_ID),
        cur_pose6=cur_pose6,
        cur_joint6=cur_joint6,
        target_pose6=last_target_pose6,
        approach_phase=int(approach_phase),
        step_try_list=list(rc.STEP_TRY_LIST_DEFAULT),
        vel_list=list(rc.MOVE_CART_VEL_LIST),
        step_scale_default=float(step_scale),
    )

    out["moved"] = moved
    out["debug"] = dbg
    out["new_phase"] = int(new_phase)

    if not moved:
        out["msg"] = "MoveCart 1-step 실패(112 포함)"
        return out

    out["ok"] = True
    out["pose_after"] = pose_after
    out["joint_after"] = joint_after

    if done:
        out["done"] = True
        out["reached_final"] = True
        out["msg"] = "최종 targetZ 도달"
    else:
        out["reached_final"] = False
        out["msg"] = "1-step 이동 완료"

    return out
