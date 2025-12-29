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

# ============================================================
# ✅ SPEED CONFIG (여기서 %로 조절)
#   - Fairino SDK에서 vel 파라미터는 보통 "퍼센트(%)"로 쓰는 경우가 많음
#   - 너가 지금 쓰는 값(20,10,5,3) 그대로 유지 + 위에서만 수정 가능하도록 분리
# ============================================================
MOVE_CART_VEL_DEFAULT = 20.0
MOVE_CART_VEL_FALLBACKS = [10.0, 5.0, 3.0]     # 실패/112 나오면 순차 적용
MOVE_CART_VEL_LIST = [MOVE_CART_VEL_DEFAULT] + MOVE_CART_VEL_FALLBACKS

MOVEJ_VEL_J6 = 10.0             # ✅ 6번 J6 회전 속도(%)
MOVEJ_BLENDT_J6 = -1.0          # -1 blocking

MOVEJ_VEL_RETURN = 15.0         # ✅ 8번 초기 복귀 속도(%)
MOVEJ_BLENDT_RETURN = -1.0      # -1 blocking

# -------------------------
# Step config
# -------------------------
STEP_SCALE_DEFAULT = 0.3     # 기본: y/z 이동 비율
X_SCALE_MULT = 2.0          # ✅ X는 y/z보다 더 많이 이동

# -------------------------
# ✅ 2-Phase approach (cmd=7,9)
# -------------------------
Z_HOLD_OFFSET_MM = 70.0     # 1단계에서 목표 Z보다 +70mm 위에서 멈춤
XY_TOL_MM = 1               # X/Y 도착 판정 허용오차(mm)
Z_TOL_MM  = 2               # Z 도착 판정 허용오차(mm)

# -------------------------
# ✅ 6번(J6 회전) config
# -------------------------
ANGLE_TO_J6_SIGN = +1.0     # angle_deg 방향이 반대면 -1.0로 바꿔
J6_MAX_STEP_DEG  = 45.0     # 안전장치: 한 번에 최대 회전각 제한

# -------------------------
# ✅ 7/9 step try config
# -------------------------
STEP_TRY_LIST_DEFAULT = [STEP_SCALE_DEFAULT, 0.05, 0.02, 0.01]

# -------------------------
# ✅ 9번(자동 시퀀스) safety
# -------------------------
AUTO_MAX_SECONDS = 60.0        # 전체 자동 시퀀스 최대 시간
AUTO_MAX_STEPS = 250           # 전체 step 이동 최대 횟수(무한루프 방지)
AUTO_ROTATE_DELAY_SEC = 0.2    # phase1 진입 후 회전 전/후 약간의 텀


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
    print(f"[RET] MoveJ(J6) errcode: {rtn}")
    return rtn


def move_to_joint(robot, joint_target6, tool, user, vel, blendT, reconnect=None):
    """
    ✅ 특정 조인트로 MoveJ 복귀 (blocking)
    """
    jt = ensure_joint6(joint_target6)

    err_j, cur_joint = safe_call(robot.GetActualJointPosDegree, flag=1, retry=1, reconnect_cb=reconnect)
    if err_j == 0:
        cur6 = ensure_joint6(cur_joint)
        print("[MOVEJ-RETURN] current joint:", fmt_joint(cur6))
        print("[MOVEJ-RETURN] target  joint:", fmt_joint(jt))
        print("[MOVEJ-RETURN] delta         :", joint_delta_str(jt, cur6))
    else:
        print("[MOVEJ-RETURN] (warn) cannot read current joint, moving anyway...")

    rtn = safe_call(
        robot.MoveJ,
        joint_pos=jt,
        tool=tool,
        user=user,
        vel=float(vel),
        blendT=float(blendT),
        retry=1,
        reconnect_cb=reconnect
    )
    print(f"[RET] MoveJ(Return) errcode: {rtn}")
    return rtn


def search_target_with_step_check(robot, cur_pose6, cur_joint6, base_target6, step_scale, reconnect=None):
    """
    ✅ target_pose True 찾기 + step_pose True도 확인 (MoveCart 112 방지용)
    ✅ 우선순위:
       1) ry 먼저 살짝
       2) rz만
       3) rx/ry/rz 조합(best score)
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

        score = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(j6, cur_joint6)))
        cand = {
            "target": cand_target,
            "step": cand_step,
            "step_joint": j6,
            "score": score,
            "d": d_tuple,
            "tries": tries
        }
        return cand, False

    # 1) ry만 먼저
    for dry in ry_list:
        cand_target = [x, y, z, rx0, ry0 + float(dry), rz0]
        cand, stop = check_candidate(cand_target, (0.0, float(dry), 0.0))
        if stop:
            return best
        if cand is not None:
            return cand

    # 2) rz만
    for drz in rz_list:
        cand_target = [x, y, z, rx0, ry0, rz0 + float(drz)]
        cand, stop = check_candidate(cand_target, (0.0, 0.0, float(drz)))
        if stop:
            return best
        if cand is not None:
            return cand

    # 3) 조합 탐색
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


# ============================================================
# ✅ 핵심: "7 한 번 분량"을 함수로 만들고
#   - 7은 이걸 1번만 수행
#   - 9는 이걸 while로 반복 수행 + phase1 진입 시 6 자동 실행
# ============================================================
def do_one_movecart_step(
    robot,
    reconnect,
    tool,
    user,
    cur_pose6,
    cur_joint6,
    target_pose,
    approach_phase,
    step_try_list,
    vel_list
):
    """
    Returns:
      moved(bool),
      pose_after(list|None),
      joint_after(list|None),
      new_phase(int),
      reached_final(bool)
    """
    target_pose = ensure_pose6(target_pose)
    z_hold = float(target_pose[2]) + float(Z_HOLD_OFFSET_MM)

    # phase별 목표 pose 구성
    if approach_phase == 0:
        phase_target_pose = target_pose[:]
        phase_target_pose[2] = z_hold
        xy_scale_candidates = [1.0]
        ori_scale_candidates = [1.0]
    else:
        phase_target_pose = target_pose[:]
        xy_scale_candidates = [0.0, 0.01, 0.02]  # 원칙 0, 필요 시 최소 보정
        ori_scale_candidates = [0.0]            # 자세 고정

    acc = 0.0
    ovl = 100.0

    moved = False
    pose_after = None
    joint_after = None
    new_phase = approach_phase
    reached_final = False

    for st in step_try_list:
        for xy_s in xy_scale_candidates:
            if approach_phase == 0:
                sx = min(1.0, st * X_SCALE_MULT)
                sy = st
                sz = st
            else:
                sx = xy_s
                sy = xy_s
                sz = st

            for ori_s in ori_scale_candidates:
                step_pose = ensure_pose6(
                    blend_pose_axis(cur_pose6, phase_target_pose, (sx, sy, sz), ori_scale=ori_s)
                )

                # ✅ step_pose HasSolution 검사
                if not has_solution(robot, step_pose, cur_joint6, reconnect=reconnect):
                    continue

                for vv in vel_list:
                    try:
                        rtn = safe_call(
                            robot.MoveCart,
                            step_pose, tool, user, float(vv), acc, ovl, -1.0, -1,
                            retry=1,
                            reconnect_cb=reconnect
                        )
                    except Exception:
                        continue

                    if rtn == 0:
                        moved = True
                        (a1, pose_after), (a2, joint_after) = read_pose_joint(robot, reconnect=reconnect)
                        if a1 != 0 or a2 != 0:
                            # 이동은 됐는데 read 실패
                            return True, None, None, new_phase, False

                        # ✅ phase 전환/완료 체크
                        if approach_phase == 0:
                            xy_ok = xy_reached(pose_after, target_pose, tol_mm=XY_TOL_MM)
                            z_ok  = z_reached(pose_after, z_hold, tol_mm=Z_TOL_MM)
                            if xy_ok and z_ok:
                                new_phase = 1
                        else:
                            z_done = z_reached(pose_after, target_pose[2], tol_mm=Z_TOL_MM)
                            if z_done:
                                reached_final = True
                                new_phase = 0  # 다음 작업 대비 리셋용(밖에서 reached_final 보고 처리)
                        return True, pose_after, joint_after, new_phase, reached_final

                    if rtn == 112:
                        time.sleep(0.05)
                        continue

                    time.sleep(0.05)

    return False, None, None, new_phase, False


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
    print(f"  7 : MoveCart 1-step (phase 진행)  (Y/Z는 *{step_scale}, X는 *{min(1.0, step_scale*X_SCALE_MULT):.3f})")
    print("  8 : ✅ 프로그램 시작(초기) 위치로 복귀 (MoveJ로 초기 joint 복귀 = J6도 원복)")
    print("  9 : ✅ 자동 시퀀스 (phase0->(자동6)->phase1 완료까지 한번에)")
    print("  q : 종료")
    print("=======================================")
    return input("입력 (1/2/3/4/5/6/7/8/9/q) > ").strip()


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

    step_scale = STEP_SCALE_DEFAULT

    last_tcp_pose = None
    last_measure = None
    last_target_pose = None
    approach_phase = 0       # 0=XY+Zhold, 1=Zdown
    reached_final = False

    # ✅ 초기(프로그램 시작) 상태 저장
    initial_joint6 = None
    try:
        (e1, _p0), (e2, j0) = read_pose_joint(robot, reconnect=reconnect)
        if e1 == 0 and e2 == 0:
            initial_joint6 = j0[:]
            print("[INIT] 프로그램 시작 초기 joint 저장 ✅")
            print("  init_joint:", fmt_joint(initial_joint6))
            print()
        else:
            print(f"[INIT-WARN] 초기 joint 저장 실패: err_p={e1}, err_j={e2}")
            print("           -> 1번을 누를 때 초기값으로 저장 시도합니다.\n")
    except Exception as e:
        print(f"[INIT-WARN] 초기 joint 저장 예외: {e}")
        print("           -> 1번을 누를 때 초기값으로 저장 시도합니다.\n")

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

                if initial_joint6 is None:
                    initial_joint6 = joint6[:]
                    print("\n[INIT] 초기 joint가 비어있어서, 지금 값을 초기 joint로 저장 ✅")
                    print("  init_joint:", fmt_joint(initial_joint6))
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
                approach_phase = 0
                reached_final = False
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
                approach_phase = 0
                reached_final = False

                drx, dry, drz = best["d"]
                print("[SEARCH] ✅ 후보 발견! (target/step 모두 solvable)")
                print(f"  tried={best['tries']}  score(delta-norm)={best['score']:.3f}")
                print(f"  dRPY(deg)=({drx:+.1f}, {dry:+.1f}, {drz:+.1f})")
                print("  new target_pose :", fmt_pose6(last_target_pose))
                print("  => last_target_pose 업데이트 ✅")
                print("  => phase reset to 0 (XY+Zhold)\n")

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

                if delta > J6_MAX_STEP_DEG:
                    delta = J6_MAX_STEP_DEG
                elif delta < -J6_MAX_STEP_DEG:
                    delta = -J6_MAX_STEP_DEG

                print("\n[ACTION] Rotate J6 by measured angle_deg")
                print(f"  angle_deg={ang:+.3f} => delta(J6)={delta:+.3f} deg")
                move_j6_by_delta_deg(robot, delta, tool, user, reconnect=reconnect)
                print()

            elif cmd == "7":
                if last_target_pose is None:
                    print("\n[WARN] target_pose 없음.\n")
                    continue
                if reached_final:
                    print("\n[INFO] 이미 최종(targetZ) 도달 상태입니다 ✅ (7 동작 안 함)\n")
                    continue

                (e1, cur_pose6), (e2, cur_joint6) = read_pose_joint(robot, reconnect=reconnect)
                if e1 != 0 or e2 != 0:
                    print(f"[FAIL] err_p={e1}, err_j={e2}\n")
                    continue

                moved, pose_after, joint_after, new_phase, done = do_one_movecart_step(
                    robot=robot,
                    reconnect=reconnect,
                    tool=tool,
                    user=user,
                    cur_pose6=cur_pose6,
                    cur_joint6=cur_joint6,
                    target_pose=last_target_pose,
                    approach_phase=approach_phase,
                    step_try_list=STEP_TRY_LIST_DEFAULT,
                    vel_list=MOVE_CART_VEL_LIST
                )

                if not moved:
                    print("\n[FAIL] MoveCart 1-step 실패 (112 포함)")
                    print("  -> step/vel 더 낮추거나, 4번 자세 탐색을 더 보수적으로\n")
                    continue

                if pose_after is not None and joint_after is not None:
                    print("\n[STATE AFTER STEP]")
                    print("  new_pose  :", fmt_pose6(pose_after))
                    print("  new_joint :", fmt_joint(joint_after))
                    print("  joint dlt :", joint_delta_str(joint_after, cur_joint6))

                if approach_phase == 0 and new_phase == 1:
                    print("\n[PHASE] ✅ phase0 완료 -> phase1 진입! (이제 Z만 내려감)\n")

                approach_phase = new_phase
                if done:
                    reached_final = True
                    approach_phase = 0
                    print("\n[DONE] ✅ 최종 targetZ 도달! (프로그램 계속)\n")

            elif cmd == "8":
                if initial_joint6 is None:
                    print("\n[WARN] 초기 joint가 저장되어 있지 않습니다.")
                    print("       - 1번을 한 번 눌러 현재값을 초기로 저장하거나, 프로그램을 재실행하세요.\n")
                    continue

                print("\n[ACTION] 초기 위치로 복귀 (MoveJ to initial_joint6)")
                rtn = move_to_joint(
                    robot,
                    joint_target6=initial_joint6,
                    tool=tool,
                    user=user,
                    vel=MOVEJ_VEL_RETURN,
                    blendT=MOVEJ_BLENDT_RETURN,
                    reconnect=reconnect
                )
                if rtn == 0:
                    approach_phase = 0
                    reached_final = False
                    print("\n[RESET] phase=0, reached_final=False 로 초기화 ✅\n")
                else:
                    print(f"\n[FAIL] 초기 복귀 실패 errcode={rtn}\n")

            elif cmd == "9":
                if last_target_pose is None:
                    print("\n[WARN] target_pose 없음. (3/4 먼저)\n")
                    continue
                if reached_final:
                    print("\n[INFO] 이미 최종(targetZ) 도달 상태입니다 ✅ (9 동작 안 함)\n")
                    continue

                print("\n[AUTO] ✅ 자동 시퀀스 시작 (phase0 -> (자동6) -> phase1 완료)")
                print(f"       - MoveCart vel(%) list = {MOVE_CART_VEL_LIST}")
                print(f"       - step list            = {STEP_TRY_LIST_DEFAULT}")
                print(f"       - AUTO_MAX_SECONDS     = {AUTO_MAX_SECONDS}")
                print(f"       - AUTO_MAX_STEPS       = {AUTO_MAX_STEPS}\n")

                rotated_once = False
                t0 = time.time()
                steps = 0

                while True:
                    if (time.time() - t0) > AUTO_MAX_SECONDS:
                        print("\n[AUTO-STOP] ❌ 시간 초과로 자동 시퀀스 중단\n")
                        break
                    if steps >= AUTO_MAX_STEPS:
                        print("\n[AUTO-STOP] ❌ step 횟수 초과로 자동 시퀀스 중단\n")
                        break
                    if reached_final:
                        print("\n[AUTO-DONE] ✅ 최종 targetZ 도달! (자동 시퀀스 종료)\n")
                        break

                    (e1, cur_pose6), (e2, cur_joint6) = read_pose_joint(robot, reconnect=reconnect)
                    if e1 != 0 or e2 != 0:
                        print(f"\n[AUTO-STOP] ❌ 상태 읽기 실패 err_p={e1}, err_j={e2}\n")
                        break

                    moved, pose_after, joint_after, new_phase, done = do_one_movecart_step(
                        robot=robot,
                        reconnect=reconnect,
                        tool=tool,
                        user=user,
                        cur_pose6=cur_pose6,
                        cur_joint6=cur_joint6,
                        target_pose=last_target_pose,
                        approach_phase=approach_phase,
                        step_try_list=STEP_TRY_LIST_DEFAULT,
                        vel_list=MOVE_CART_VEL_LIST
                    )

                    steps += 1

                    if not moved:
                        print("\n[AUTO-STOP] ❌ MoveCart step 실패(112 포함)로 자동 시퀀스 중단")
                        print("          -> 4번 자세 탐색(자세 변화량 더 작게) / vel/step 더 보수적으로\n")
                        break

                    # 로그(간단히)
                    if pose_after is not None:
                        print(f"[AUTO] step#{steps:03d}  phase {approach_phase} -> {new_phase}  pose={fmt_pose6(pose_after)}")

                    # phase 갱신
                    phase_changed_to_1 = (approach_phase == 0 and new_phase == 1)
                    approach_phase = new_phase

                    # ✅ phase1 진입 시 "자동 6(회전)" 1회 실행
                    if phase_changed_to_1 and (not rotated_once):
                        print("\n[AUTO] ✅ phase1 진입 감지 -> 자동으로 6(J6 회전) 1회 실행")
                        time.sleep(AUTO_ROTATE_DELAY_SEC)

                        if last_measure is None:
                            print("[AUTO-WARN] last_measure 없음(2번 미실행?) -> 회전 스킵하고 Zdown 진행\n")
                        else:
                            ang = float(last_measure.get("angle_deg", 0.0))
                            delta = ANGLE_TO_J6_SIGN * ang
                            if delta > J6_MAX_STEP_DEG:
                                delta = J6_MAX_STEP_DEG
                            elif delta < -J6_MAX_STEP_DEG:
                                delta = -J6_MAX_STEP_DEG

                            print(f"[AUTO] angle_deg={ang:+.3f} => delta(J6)={delta:+.3f} deg")
                            move_j6_by_delta_deg(robot, delta, tool, user, reconnect=reconnect)
                            rotated_once = True
                            time.sleep(AUTO_ROTATE_DELAY_SEC)
                            print("[AUTO] ✅ 회전 완료. Zdown 계속 진행\n")

                    if done:
                        reached_final = True
                        approach_phase = 0
                        print("\n[AUTO] ✅ 최종 targetZ 도달! (프로그램 계속)\n")
                        # loop will break next iteration due to reached_final

                print("[AUTO] 자동 시퀀스 종료. 메뉴로 돌아갑니다.\n")

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
