import sys
import time
import math

from box_test_3 import main as measure_main

# ✅ Robot.cp39-win_amd64.pyd 폴더를 Python 경로에 추가
sys.path.insert(
    0,
    r"C:\Users\rhdeh\ws_job_msis\amr_project\driver\fairino-python-sdk-main\windows\fairino\build\lib.win-amd64-cpython-39"
)
import Robot

ROBOT_IP = "192.168.0.15"

# -------------------------
# Step config
# -------------------------
STEP_SCALE_DEFAULT = 0.3     # 기본: y/z 10% 이동
X_SCALE_MULT = 2.0          # ✅ X는 y/z보다 더 많이 이동 (예: 3배 => x는 30%)

# -------------------------
# ✅ 2-Phase approach (cmd=7)
# -------------------------
Z_HOLD_OFFSET_MM = 70.0     # 1단계에서 목표 Z보다 +50mm 위에서 멈춤
XY_TOL_MM = 1             # X/Y 도착 판정 허용오차(mm)
Z_TOL_MM  = 2           # Z 홀드 도착 판정 허용오차(mm)

# -------------------------
# ✅ 6번(J6 회전) config
# -------------------------
ANGLE_TO_J6_SIGN = +1.0     # angle_deg 방향이 반대면 -1.0로 바꿔
J6_MAX_STEP_DEG  = 45.0     # 안전장치: 한 번에 최대 회전각 제한
MOVEJ_VEL_J6     = 10.0     # MoveJ 속도(%)
MOVEJ_BLENDT_J6  = -1.0     # -1 blocking


# -------------------------
# Utils
# -------------------------
def fmt_pose6(pose):
    if not isinstance(pose, (list, tuple)) or len(pose) < 6:
        return str(pose)
    x, y, z, rx, ry, rz = pose[:6]
    return f"[x,y,z,rx,ry,rz]=[{x:.3f}, {y:.3f}, {z:.3f}, {rx:.3f}, {ry:.3f}, {rz:.3f}]"


def fmt_joint(j):
    if not isinstance(j, (list, tuple)):
        return str(j)
    return "[" + ", ".join(f"{float(v):.3f}" for v in j[:6]) + "]"


def ensure_pose6(p):
    if not isinstance(p, (list, tuple)):
        raise ValueError(f"pose is not list/tuple: {type(p)}")
    if len(p) < 6:
        raise ValueError(f"pose length < 6: {len(p)}")
    return [float(x) for x in p[:6]]


def ensure_joint6(j):
    if not isinstance(j, (list, tuple)):
        raise ValueError(f"joint is not list/tuple: {type(j)}")
    if len(j) < 6:
        raise ValueError(f"joint length < 6: {len(j)}")
    return [float(x) for x in j[:6]]


def joint_delta_str(j_new, j_old):
    if j_new is None or j_old is None:
        return "(delta N/A)"
    d = [float(a) - float(b) for a, b in zip(j_new[:6], j_old[:6])]
    return "[" + ", ".join(f"{v:+.3f}" for v in d) + "]"


def build_target_pose(cur_pose, move_res):
    x, y, z, rx, ry, rz = ensure_pose6(cur_pose)
    dx = float(move_res["move_x_mm"])
    dy = float(move_res["move_y_mm"])
    dz = float(move_res["move_z_mm"])
    return [x + dx, y + dy, z + dz, rx, ry, rz]


def blend_pose(cur_pose, target_pose, scale):
    cur = ensure_pose6(cur_pose)
    tgt = ensure_pose6(target_pose)
    s = float(scale)
    return [cur[i] + (tgt[i] - cur[i]) * s for i in range(6)]


def blend_pose_axis(cur_pose, target_pose, scale_xyz, ori_scale=None):
    """
    ✅ 축별 다른 비율로 step 이동
    - scale_xyz=(sx,sy,sz)
    - target을 넘어가지 않도록 clamp
    - ori_scale: rx/ry/rz 스케일(없으면 sy 사용)
    """
    cur = ensure_pose6(cur_pose)
    tgt = ensure_pose6(target_pose)

    sx, sy, sz = [float(v) for v in scale_xyz]
    if ori_scale is None:
        ori_scale = sy
    else:
        ori_scale = float(ori_scale)

    out = cur[:]  # copy

    # XYZ
    for i, s in zip([0, 1, 2], [sx, sy, sz]):
        delta = tgt[i] - cur[i]
        step = delta * s

        # overshoot 방지
        if delta >= 0:
            out[i] = min(cur[i] + step, tgt[i])
        else:
            out[i] = max(cur[i] + step, tgt[i])

    # RPY (서서히)
    for i in [3, 4, 5]:
        out[i] = cur[i] + (tgt[i] - cur[i]) * ori_scale

    return out


def xy_reached(cur_pose6, target_pose6, tol_mm=2.0):
    cur = ensure_pose6(cur_pose6)
    tgt = ensure_pose6(target_pose6)
    return (abs(cur[0] - tgt[0]) <= tol_mm) and (abs(cur[1] - tgt[1]) <= tol_mm)


def z_reached(cur_pose6, z_target, tol_mm=2.0):
    cur = ensure_pose6(cur_pose6)
    return abs(cur[2] - float(z_target)) <= tol_mm


def connect_robot(ip):
    print("[INFO] Connecting to Fairino...")
    rb = Robot.RPC(ip)
    print(rb)
    print("[INFO] Robot connected ✅\n")
    return rb


def safe_call(fn, *args, retry=1, sleep_sec=0.25, reconnect_cb=None, **kwargs):
    last_e = None
    for k in range(retry + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_e = e
            msg = str(e)
            if ("timed out" in msg.lower()) or ("实时数据失败" in msg) or ("timeout" in msg.lower()):
                print(f"[WARN] RPC timeout-like error: {e}")
                if reconnect_cb is not None:
                    try:
                        reconnect_cb()
                    except Exception as e2:
                        print(f"[WARN] reconnect failed: {e2}")
            if k < retry:
                time.sleep(sleep_sec)
                continue
            raise last_e


def read_pose_joint(robot, reconnect=None):
    err_p, pose = safe_call(robot.GetActualTCPPose, flag=1, retry=1, reconnect_cb=reconnect)
    err_j, joint = safe_call(robot.GetActualJointPosDegree, flag=1, retry=1, reconnect_cb=reconnect)
    if err_p != 0 or err_j != 0:
        return (err_p, None), (err_j, None)
    return (0, ensure_pose6(pose)), (0, ensure_joint6(joint))


def has_solution(robot, pose6, cur_joint6, reconnect=None):
    err, ok = safe_call(robot.GetInverseKinHasSolution, 0, pose6, cur_joint6, retry=1, reconnect_cb=reconnect)
    if err != 0:
        return False
    return bool(ok)


def get_ik(robot, pose6, cur_joint6, reconnect=None):
    err, j = safe_call(robot.GetInverseKinRef, 0, pose6, cur_joint6, retry=1, reconnect_cb=reconnect)
    if err != 0:
        return None
    return ensure_joint6(j)


def joint_delta_norm(j_sol6, j_ref6):
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(j_sol6, j_ref6)))


def move_j6_by_delta_deg(robot, delta_deg, tool, user, reconnect=None):
    """
    ✅ 6번: box_test_3에서 나온 angle_deg 만큼 J6만 상대 회전 (MoveJ)
    """
    err_j, cur_joint = safe_call(robot.GetActualJointPosDegree, flag=1, retry=1, reconnect_cb=reconnect)
    if err_j != 0:
        print(f"[FAIL] GetActualJointPosDegree err={err_j}")
        return err_j

    j_now6 = ensure_joint6(cur_joint)
    j_tgt = list(j_now6)
    j_tgt[5] = float(j_tgt[5]) + float(delta_deg)

    print(f"[MOVEJ-J6] J6: {j_now6[5]:.3f} -> {j_tgt[5]:.3f} (delta {delta_deg:+.3f} deg)")

    rtn = safe_call(
        robot.MoveJ,
        joint_pos=j_tgt,
        tool=tool,
        user=user,
        vel=float(MOVEJ_VEL_J6),
        blendT=float(MOVEJ_BLENDT_J6),
        retry=1,
        reconnect_cb=reconnect
    )
    print(f"[RET] MoveJ errcode: {rtn}")
    return rtn


def search_target_with_step_check(robot, cur_pose6, cur_joint6, base_target6, step_scale, reconnect=None):
    """
    ✅ target_pose True 찾기 + step_pose True도 확인 (MoveCart 112 방지용)
    ✅ 우선순위 변경:
       1) ry 먼저 살짝 건드림 (rz 고정)
       2) 그 다음 rz만
       3) 마지막에 rx/ry/rz 조합(best score)
    """
    x, y, z, rx0, ry0, rz0 = base_target6

    rz_list = [0, 1, -1, 2, -2, 3, -3, 5, -5, 8, -8, 12, -12, 15, -15, 20, -20, 30, -30]
    rx_list = [0, 1, -1, 2, -2, 3, -3, 5, -5, 8, -8, 12, -12, 15, -15]
    ry_list = [0, 1, -1, 2, -2, 3, -3, 5, -5, 8, -8, 12, -12, 15, -15]

    best = None
    tries = 0
    t0 = time.time()

    def check_candidate(cand_target, d_tuple):
        nonlocal tries
        tries += 1

        # 시간 제한
        if (time.time() - t0) > 8.0:
            return None, True  # stop

        if not has_solution(robot, cand_target, cur_joint6, reconnect=reconnect):
            return None, False

        cand_step = ensure_pose6(blend_pose(cur_pose6, cand_target, step_scale))
        if not has_solution(robot, cand_step, cur_joint6, reconnect=reconnect):
            return None, False

        j6 = get_ik(robot, cand_step, cur_joint6, reconnect=reconnect)
        if j6 is None:
            return None, False

        score = joint_delta_norm(j6, cur_joint6)
        cand = {
            "target": cand_target,
            "step": cand_step,
            "step_joint": j6,
            "score": score,
            "d": d_tuple,
            "tries": tries
        }
        return cand, False

    # -----------------------------
    # 1) ✅ ry만 먼저 (rz 고정)
    # -----------------------------
    for dry in ry_list:
        cand_target = [x, y, z, rx0, ry0 + float(dry), rz0]
        cand, stop = check_candidate(cand_target, (0.0, float(dry), 0.0))
        if stop:
            return best
        if cand is not None:
            return cand

    # -----------------------------
    # 2) 그 다음 rz만 (기존 1단계였던 것)
    # -----------------------------
    for drz in rz_list:
        cand_target = [x, y, z, rx0, ry0, rz0 + float(drz)]
        cand, stop = check_candidate(cand_target, (0.0, 0.0, float(drz)))
        if stop:
            return best
        if cand is not None:
            return cand

    # -----------------------------
    # 3) 마지막: rx/ry/rz 조합 (best score)
    # -----------------------------
    for drx in rx_list:
        for dry in ry_list:
            for drz in rz_list:
                if tries > 900 or (time.time() - t0) > 8.0:
                    return best

                cand_target = [x, y, z, rx0 + float(drx), ry0 + float(dry), rz0 + float(drz)]
                cand, stop = check_candidate(cand_target, (float(drx), float(dry), float(drz)))
                if stop:
                    return best
                if cand is None:
                    continue

                if (best is None) or (cand["score"] < best["score"]):
                    best = cand

    return best


def prompt_menu(step_scale):
    print("=======================================")
    print("무슨 기능을 할까요?")
    print("  1 : 현재 tcp_pose + joint(deg) 가져오기/저장")
    print("  2 : box_test_3 측정 실행/저장 (moveXYZ/angle)")
    print("  3 : target_pose 생성/저장 (1+2)")
    print("  4 : IK 가능/불가능 확인 (HasSolution)")
    print("      - False면 rx/ry/rz를 조금씩 바꿔서 True 되는 target_pose 탐색")
    print("      - ✅ step_pose(현재→target*step)도 True인 후보만 채택")
    print("  5 : IK 해(joint 솔루션) 출력 (현재 last_target_pose 기준)")
    print("  6 : ✅ box_test_3 angle_deg 만큼 J6 회전 (MoveJ)")
    print(f"  7 : MoveCart 이동 (2단계: XY+Zhold → Zdown)  (Y/Z는 *{step_scale}, X는 *{min(1.0, step_scale*X_SCALE_MULT):.3f})")
    print("  q : 종료")
    print("=======================================")
    return input("입력 (1/2/3/4/5/6/7/q) > ").strip()


def main():
    robot = connect_robot(ROBOT_IP)

    def reconnect():
        nonlocal robot
        try:
            robot.CloseRPC()
        except Exception:
            pass
        time.sleep(0.3)
        robot = connect_robot(ROBOT_IP)

    tool = 0
    user = 0

    vel_default = 20.0
    acc = 0.0
    ovl = 100.0

    step_scale = STEP_SCALE_DEFAULT

    last_tcp_pose = None
    last_measure = None
    last_target_pose = None  # 최종 목표
    approach_phase = 0       # 0: XY맞추고 Z는 (targetZ+50)에서 멈춤, 1: Z만 targetZ로 내려감

    try:
        while True:
            cmd = prompt_menu(step_scale)

            if cmd == "1":
                print("\n[ACTION] GetActualTCPPose / Joint...")
                (e1, pose6), (e2, joint6) = read_pose_joint(robot, reconnect=reconnect)
                if e1 != 0 or e2 != 0:
                    print(f"[FAIL] err_p={e1}, err_j={e2}\n")
                    continue

                last_tcp_pose = pose6
                print("[OK] 현재 상태 저장 ✅")
                print("tcp_pose :", fmt_pose6(pose6))
                print("joint6   :", fmt_joint(joint6))
                print()

            elif cmd == "2":
                print("\n[ACTION] box_test_3 측정 시작...")
                try:
                    res = measure_main()
                except Exception as e:
                    print(f"[ERROR] box_test_3 실행 예외: {e}\n")
                    continue
                if res is None:
                    print("[FAIL] 측정 실패(None)\n")
                    continue
                last_measure = res
                print("[OK] 측정 결과 저장 ✅")
                print(f"moveXYZ(mm) = ({float(res['move_x_mm']):+.1f}, {float(res['move_y_mm']):+.1f}, {float(res['move_z_mm']):+.1f})")
                print(f"angle(deg)  = {float(res.get('angle_deg', 0.0)):+.2f}\n")

            elif cmd == "3":
                if last_tcp_pose is None or last_measure is None:
                    print("\n[WARN] 1,2번 먼저.\n")
                    continue
                last_target_pose = build_target_pose(last_tcp_pose, last_measure)
                approach_phase = 0  # ✅ 새 목표면 다시 1단계부터
                print("\n[OK] target_pose 생성/저장 ✅")
                print("current_pose :", fmt_pose6(last_tcp_pose))
                print("target_pose  :", fmt_pose6(last_target_pose))
                print("phase        : 0 (XY+Zhold)\n")

            elif cmd == "4":
                if last_target_pose is None:
                    print("\n[WARN] 3번으로 target_pose 먼저 생성.\n")
                    continue

                (e1, cur_pose6), (e2, cur_joint6) = read_pose_joint(robot, reconnect=reconnect)
                if e1 != 0 or e2 != 0:
                    print(f"[FAIL] err_p={e1}, err_j={e2}\n")
                    continue

                target = ensure_pose6(last_target_pose)

                ok = has_solution(robot, target, cur_joint6, reconnect=reconnect)
                if ok:
                    print(f"[RESULT] HasSolution(target_pose) = True ✅\n")
                    continue

                print("[RESULT] HasSolution(target_pose) = False ❌")
                print("  -> rx/ry/rz를 조금씩 바꿔서 True(+step True) 되는 target_pose를 탐색합니다...")

                best = search_target_with_step_check(
                    robot=robot,
                    cur_pose6=cur_pose6,
                    cur_joint6=cur_joint6,
                    base_target6=target,
                    step_scale=step_scale,
                    reconnect=reconnect
                )

                if best is None:
                    print("[SEARCH] ❌ 실패: 근처 rx/ry/rz 변화로 True+step True 후보를 못 찾음\n")
                    continue

                last_target_pose = best["target"]
                approach_phase = 0  # ✅ 목표 pose가 바뀌었으니 다시 1단계부터
                drx, dry, drz = best["d"]

                print("[SEARCH] ✅ 후보 발견! (target/step 모두 solvable)")
                print(f"  tried={best['tries']}  score(delta-norm)={best['score']:.3f}")
                print(f"  dRPY(deg)=({drx:+.1f}, {dry:+.1f}, {drz:+.1f})")
                print("  new target_pose :", fmt_pose6(last_target_pose))
                print("  step IK joint6  :", fmt_joint(best["step_joint"]))
                print("  => last_target_pose 업데이트 ✅")
                print("  => phase reset to 0 (XY+Zhold)\n")

                ok2 = has_solution(robot, ensure_pose6(last_target_pose), cur_joint6, reconnect=reconnect)
                print(f"[RECHECK] updated target_pose HasSolution = {ok2}\n")

            elif cmd == "5":
                if last_target_pose is None:
                    print("\n[WARN] target_pose 없음.\n")
                    continue

                err_j, cur_joint = safe_call(robot.GetActualJointPosDegree, flag=1, retry=1, reconnect_cb=reconnect)
                if err_j != 0:
                    print(f"[FAIL] GetActualJointPosDegree err={err_j}\n")
                    continue

                cur_joint6 = ensure_joint6(cur_joint)
                target = ensure_pose6(last_target_pose)

                if not has_solution(robot, target, cur_joint6, reconnect=reconnect):
                    print("[RESULT] ❌ IK 해 없음 (현재 target_pose 기준)\n")
                    continue

                j6 = get_ik(robot, target, cur_joint6, reconnect=reconnect)
                print("[OK] IK Joint Solution ✅")
                print("target_pose   :", fmt_pose6(target))
                print("joint_sol6    :", fmt_joint(j6))
                print()

            elif cmd == "6":
                if last_measure is None:
                    print("\n[WARN] 2번(box_test_3 측정) 먼저 실행해야 angle_deg가 있어요.\n")
                    continue

                ang = float(last_measure.get("angle_deg", 0.0))
                delta = ANGLE_TO_J6_SIGN * ang

                # 안전장치 clamp
                if delta > J6_MAX_STEP_DEG:
                    delta = J6_MAX_STEP_DEG
                elif delta < -J6_MAX_STEP_DEG:
                    delta = -J6_MAX_STEP_DEG

                print("\n[ACTION] Rotate J6 by measured angle_deg (from box_test_3)")
                print(f"  angle_deg={ang:+.3f}  sign={ANGLE_TO_J6_SIGN:+.1f}  => delta(J6)={delta:+.3f} deg")

                move_j6_by_delta_deg(robot, delta, tool, user, reconnect=reconnect)
                print()

            elif cmd == "7":
                if last_target_pose is None:
                    print("\n[WARN] target_pose 없음.\n")
                    continue

                # ✅ 이동 전 현재 상태 (POSE + JOINT) 출력
                (e1, cur_pose6), (e2, cur_joint6) = read_pose_joint(robot, reconnect=reconnect)
                if e1 != 0 or e2 != 0:
                    print(f"[FAIL] err_p={e1}, err_j={e2}\n")
                    continue

                target_pose = ensure_pose6(last_target_pose)
                z_hold = float(target_pose[2]) + float(Z_HOLD_OFFSET_MM)

                print("\n[STATE BEFORE]")
                print("  phase    :", approach_phase, "(0=XY+Zhold, 1=Zdown)")
                print("  cur_pose :", fmt_pose6(cur_pose6))
                print("  cur_joint:", fmt_joint(cur_joint6))
                print("  tgt_pose :", fmt_pose6(target_pose))
                print(f"  z_hold   : {z_hold:.3f} (targetZ + {Z_HOLD_OFFSET_MM}mm)")

                vel_try_list = [vel_default, 10.0, 5.0, 3.0]
                step_try_list = [step_scale, 0.05, 0.02, 0.01]

                moved = False
                joint_before = cur_joint6[:]  # ✅ delta 비교용

                # -------------------------
                # ✅ Phase별 목표 pose 구성
                # -------------------------
                if approach_phase == 0:
                    # 1단계: X/Y는 목표로, Z는 목표보다 위(z_hold)까지만
                    phase_target_pose = target_pose[:]
                    phase_target_pose[2] = z_hold
                    # X/Y는 target에 최대한 가고, Z는 z_hold까지 접근
                    xy_scale_candidates = [1.0]  # phase0는 XY 적극적으로
                    ori_scale_candidates = [1.0] # 기존처럼 함께
                else:
                    # 2단계: Z만 내려감 (가능한 한 XY는 안 움직이기)
                    phase_target_pose = target_pose[:]
                    xy_scale_candidates = [0.0, 0.01, 0.02]  # ✅ 원칙은 0, IK 안되면 최소 보정
                    ori_scale_candidates = [0.0]            # ✅ 자세 고정(원하면 0.02 등으로 바꿔도 됨)

                for st in step_try_list:
                    # 축 스케일 구성
                    # - phase0: 기존 방식(X는 더 크게), Y/Z는 st
                    # - phase1: X/Y는 0(원칙), Z만 st
                    for xy_s in xy_scale_candidates:
                        if approach_phase == 0:
                            sx = min(1.0, st * X_SCALE_MULT)
                            sy = st
                            sz = st
                        else:
                            sx = xy_s
                            sy = xy_s
                            sz = st

                        # ori_scale
                        for ori_s in ori_scale_candidates:
                            step_pose = ensure_pose6(
                                blend_pose_axis(cur_pose6, phase_target_pose, (sx, sy, sz), ori_scale=ori_s)
                            )

                            print("\n[PREVIEW] phase-step")
                            print(f"  phase      = {approach_phase}")
                            print(f"  scaleXYZ   = (sx={sx:.3f}, sy={sy:.3f}, sz={sz:.3f})  ori_scale={ori_s:.3f}")
                            print("  cur_pose   :", fmt_pose6(cur_pose6))
                            print("  tgt_phase  :", fmt_pose6(phase_target_pose))
                            print("  step_pose  :", fmt_pose6(step_pose))

                            # ✅ step_pose HasSolution 검사
                            if not has_solution(robot, step_pose, cur_joint6, reconnect=reconnect):
                                print(f"[PRECHECK] step_pose HasSolution=False -> skip (st={st}, sx={sx:.3f}, sy={sy:.3f}, sz={sz:.3f})")
                                continue

                            for vv in vel_try_list:
                                print("\n[ACTION] MoveCart 시도")
                                print(f"  phase={approach_phase}  vel={vv}  st={st}  sx/sy/sz=({sx:.3f},{sy:.3f},{sz:.3f})")
                                try:
                                    rtn = safe_call(
                                        robot.MoveCart,
                                        step_pose, tool, user, vv, acc, ovl, -1.0, -1,
                                        retry=1,
                                        reconnect_cb=reconnect
                                    )
                                except Exception as e:
                                    print(f"[ERROR] MoveCart 예외: {e}")
                                    continue

                                print(f"[RET] MoveCart errcode: {rtn}")

                                if rtn == 0:
                                    print("[OK] MoveCart 성공 ✅")

                                    # ✅ 이동 후 현재 joint/pose 다시 읽어서 변화 확인
                                    (a1, pose_after), (a2, joint_after) = read_pose_joint(robot, reconnect=reconnect)
                                    if a1 == 0 and a2 == 0:
                                        print("\n[STATE AFTER]")
                                        print("  new_pose  :", fmt_pose6(pose_after))
                                        print("  new_joint :", fmt_joint(joint_after))
                                        print("  joint dlt :", joint_delta_str(joint_after, joint_before))

                                        # ✅ Phase 전환 조건
                                        if approach_phase == 0:
                                            # XY는 target 기준, Z는 z_hold 기준
                                            xy_ok = xy_reached(pose_after, target_pose, tol_mm=XY_TOL_MM)
                                            z_ok  = z_reached(pose_after, z_hold, tol_mm=Z_TOL_MM)

                                            print(f"  check     : XY_ok={xy_ok} (tol={XY_TOL_MM}mm), Zhold_ok={z_ok} (tol={Z_TOL_MM}mm)")
                                            if xy_ok and z_ok:
                                                approach_phase = 1
                                                print("\n[PHASE] ✅ XY 정렬 + Z홀드 도착!")
                                                print("        -> 이제 7번을 다시 누르면 Z만 내려갑니다 (phase=1)\n")
                                        else:
                                            # phase1은 targetZ 도착하면 phase를 0으로 리셋(다음 작업 대비)
                                            z_done = z_reached(pose_after, target_pose[2], tol_mm=Z_TOL_MM)
                                            print(f"  check     : Z_target_ok={z_done} (tol={Z_TOL_MM}mm)")
                                            if z_done:
                                                print("\n[PHASE] ✅ Z 하강 완료! (targetZ 도착)")
                                                return  # ✅ 여기서 main() 종료 -> finally에서 CloseRPC 실행됨
                                                


                                    else:
                                        print(f"[WARN] after-state read failed: err_p={a1}, err_j={a2}")

                                    print()
                                    moved = True
                                    break

                                if rtn == 112:
                                    print("[WARN] err=112 (경로 생성 실패) -> vel/step 더 낮춰 재시도")
                                    time.sleep(0.2)
                                    continue

                                print("[WARN] MoveCart 실패 -> 다른 파라미터로 재시도\n")
                                time.sleep(0.2)

                            if moved:
                                break
                        if moved:
                            break
                    if moved:
                        break

                if not moved:
                    print("\n[FAIL] MoveCart: step/vel을 낮춰도 경로 생성 실패(112 포함)")
                    print("  -> 해결책:")
                    print("     1) 4번에서 자세 변화량 더 줄이기 (rz 1도 단위만 허용)")
                    print("     2) XYZ를 더 잘게 쪼개서 접근 (특히 Y/Z 큰 이동은 0.02~0.05 추천)")
                    print("     3) 가능하면 MoveJ(조인트)로 중간자세 만든 다음 다시 MoveCart\n")

            elif cmd.lower() == "q":
                print("\n[EXIT] 종료합니다.")
                break

            else:
                print("\n[WARN] 잘못된 입력입니다.\n")

            time.sleep(0.05)

    finally:
        try:
            robot.CloseRPC()
        except Exception:
            pass


if __name__ == "__main__":
    main()
