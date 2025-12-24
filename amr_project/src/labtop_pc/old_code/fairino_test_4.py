import sys
import time

from box_test_3 import main as measure_main

# ✅ Robot.cp39-win_amd64.pyd 폴더를 Python 경로에 추가
sys.path.insert(
    0,
    r"C:\Users\rhdeh\ws_job_msis\amr_project\driver\fairino-python-sdk-main\windows\fairino\build\lib.win-amd64-cpython-39"
)

# ✅ "Robot 모듈(.pyd)" 직접 import
import Robot


ROBOT_IP = "192.168.58.3"

# ✅ 0.1이면 1/10씩만 이동, 1.0이면 target까지 한 번에 이동
STEP_SCALE = 0.1


# -------------------------
# Utils
# -------------------------
def fmt_pose6(pose):
    if not isinstance(pose, (list, tuple)) or len(pose) < 6:
        return str(pose)
    x, y, z, rx, ry, rz = pose[:6]
    return f"[x,y,z,rx,ry,rz]=[{x:.3f}, {y:.3f}, {z:.3f}, {rx:.3f}, {ry:.3f}, {rz:.3f}]"


def fmt_joint6(j):
    if not isinstance(j, (list, tuple)) or len(j) < 6:
        return str(j)
    j1, j2, j3, j4, j5, j6 = j[:6]
    return f"[j1..j6]=[{j1:.3f}, {j2:.3f}, {j3:.3f}, {j4:.3f}, {j5:.3f}, {j6:.3f}]"


def ensure_pose6(p):
    if not isinstance(p, (list, tuple)):
        raise ValueError(f"pose is not list/tuple: {type(p)}")
    if len(p) < 6:
        raise ValueError(f"pose length < 6: {len(p)}")
    return list(p[:6])


def ensure_joint6(j):
    if not isinstance(j, (list, tuple)):
        raise ValueError(f"joint is not list/tuple: {type(j)}")
    if len(j) < 6:
        raise ValueError(f"joint length < 6: {len(j)}")
    return list(j[:6])


def build_target_pose(cur_pose, move_res):
    """angle은 적용 안 하고 XYZ만 반영해 target_pose 생성 (100% 기준)"""
    x, y, z, rx, ry, rz = ensure_pose6(cur_pose)
    dx = float(move_res["move_x_mm"])
    dy = float(move_res["move_y_mm"])
    dz = float(move_res["move_z_mm"])
    return [x + dx, y + dy, z + dz, rx, ry, rz]


def blend_pose(cur_pose, target_pose, scale):
    """현재 -> 타겟 방향으로 scale만큼만 이동한 중간 pose"""
    cur = ensure_pose6(cur_pose)
    tgt = ensure_pose6(target_pose)
    out = []
    for i in range(6):
        out.append(cur[i] + (tgt[i] - cur[i]) * float(scale))
    return out


def get_actual_joint_deg(robot, flag=1):
    """✅ 현재 joint 각도(deg) 읽기"""
    err, joint_pos = robot.GetActualJointPosDegree(flag=flag)
    return err, joint_pos


def prompt_menu():
    print("=======================================")
    print("무슨 기능을 할까요?")
    print("  1 : 현재 tcp_pose 가져오기/저장 (GetActualTCPPose)")
    print("  2 : box_test_3 측정 실행/저장 (moveXYZ/angle)")
    print("  3 : target_pose 생성/저장 (1+2)")
    print("  4 : IK 가능/불가능 확인 (HasSolution)")
    print("  5 : IK 해(joint 솔루션) 계산해서 출력")
    print(f"  6 : MoveL로 이동 (현재→target * {STEP_SCALE})")
    print(f"  7 : MoveCart로 이동 (현재→target * {STEP_SCALE})")
    print("  q : 종료")
    print("=======================================")
    return input("입력 (1/2/3/4/5/6/7/q) > ").strip()


def main():
    print("[INFO] Connecting to Fairino...")
    robot = Robot.RPC(ROBOT_IP)
    print("[INFO] Robot connected ✅\n")

    # -------------------------
    # Motion params (원하면 바꿔)
    # -------------------------
    tool = 0
    user = 0
    vel = 20.0     # 0~100 (%)
    acc = 0.0
    ovl = 100.0
    blendT = -1.0  # blocking
    blendR = -1.0  # blocking

    last_tcp_pose = None
    last_measure = None
    last_target_pose = None

    try:
        while True:
            cmd = prompt_menu()

            # 1) 현재 TCP pose 저장
            if cmd == "1":
                print("\n[ACTION] GetActualTCPPose(flag=1)...")
                try:
                    err, tcp_pose = robot.GetActualTCPPose(flag=1)
                except Exception as e:
                    print(f"[ERROR] GetActualTCPPose 예외: {e}\n")
                    continue

                if err != 0:
                    print(f"[FAIL] GetActualTCPPose err={err}, pose={tcp_pose}\n")
                    continue

                last_tcp_pose = tcp_pose
                print("[OK] tcp_pose 저장 ✅")
                print(fmt_pose6(last_tcp_pose))
                print()

            # 2) box_test_3 측정 저장
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
                mx = float(res["move_x_mm"])
                my = float(res["move_y_mm"])
                mz = float(res["move_z_mm"])
                ang = float(res.get("angle_deg", 0.0))

                print("[OK] 측정 결과 저장 ✅")
                print(f"moveXYZ(mm) = ({mx:+.1f}, {my:+.1f}, {mz:+.1f})")
                print(f"angle(deg)  = {ang:+.2f}  (saved only)\n")

            # 3) target_pose 생성 저장
            elif cmd == "3":
                if last_tcp_pose is None:
                    print("\n[WARN] 1번 먼저.\n")
                    continue
                if last_measure is None:
                    print("\n[WARN] 2번 먼저.\n")
                    continue

                last_target_pose = build_target_pose(last_tcp_pose, last_measure)
                print("\n[OK] target_pose 생성/저장 ✅")
                print("current_pose :", fmt_pose6(last_tcp_pose))
                print("target_pose  :", fmt_pose6(last_target_pose))
                print()

            # 4) IK 가능/불가능만
            elif cmd == "4":
                if last_target_pose is None:
                    print("\n[WARN] 3번으로 target_pose 먼저 생성.\n")
                    continue

                # ✅ 현재 joint(deg) 읽기
                try:
                    err_j, cur_joint = get_actual_joint_deg(robot, flag=1)
                except Exception as e:
                    print(f"[ERROR] GetActualJointPosDegree 예외: {e}\n")
                    continue

                if err_j != 0:
                    print(f"[FAIL] GetActualJointPosDegree err={err_j}, joint={cur_joint}\n")
                    continue

                cur_joint = ensure_joint6(cur_joint)
                target = ensure_pose6(last_target_pose)

                try:
                    err, result = robot.GetInverseKinHasSolution(0, target, cur_joint)
                except Exception as e:
                    print(f"[ERROR] GetInverseKinHasSolution 예외: {e}\n")
                    continue

                if err != 0:
                    print(f"[FAIL] HasSolution err={err}, result={result}\n")
                else:
                    print(f"[RESULT] HasSolution = {bool(result)}\n")

            # 5) IK 해(joint 솔루션) 출력
            elif cmd == "5":
                if last_target_pose is None:
                    print("\n[WARN] 3번으로 target_pose 먼저 생성.\n")
                    continue

                # ✅ 현재 joint(deg) 읽기
                try:
                    err_j, cur_joint = get_actual_joint_deg(robot, flag=1)
                except Exception as e:
                    print(f"[ERROR] GetActualJointPosDegree 예외: {e}\n")
                    continue

                if err_j != 0:
                    print(f"[FAIL] GetActualJointPosDegree err={err_j}, joint={cur_joint}\n")
                    continue

                cur_joint = ensure_joint6(cur_joint)
                target = ensure_pose6(last_target_pose)

                # 존재여부 확인
                try:
                    err_sol, has_sol = robot.GetInverseKinHasSolution(0, target, cur_joint)
                except Exception as e:
                    print(f"[ERROR] GetInverseKinHasSolution 예외: {e}\n")
                    continue

                if err_sol != 0:
                    print(f"[FAIL] HasSolution err={err_sol}, result={has_sol}\n")
                    continue
                if not bool(has_sol):
                    print("[RESULT] ❌ IK 해 없음: target 도달 불가\n")
                    continue

                print("\n[ACTION] GetInverseKinRef(type=0, target, current_joint)...")
                try:
                    err_ik, joint_sol = robot.GetInverseKinRef(0, target, cur_joint)
                except Exception as e:
                    print(f"[ERROR] GetInverseKinRef 예외: {e}\n")
                    continue

                if err_ik != 0:
                    print(f"[FAIL] GetInverseKinRef err={err_ik}, joint_sol={joint_sol}\n")
                    continue

                joint_sol = ensure_joint6(joint_sol)
                dj = [joint_sol[i] - cur_joint[i] for i in range(6)]

                print("[OK] IK Joint Solution ✅")
                print("current_joint:", fmt_joint6(cur_joint))
                print("joint_sol    :", fmt_joint6(joint_sol))
                print(f"delta(deg)   : ({dj[0]:+.3f}, {dj[1]:+.3f}, {dj[2]:+.3f}, {dj[3]:+.3f}, {dj[4]:+.3f}, {dj[5]:+.3f})\n")

            # 6) MoveL 이동
            elif cmd == "6":
                if last_target_pose is None:
                    print("\n[WARN] 3번으로 target_pose 먼저 생성.\n")
                    continue

                # 현재 TCP 읽고 step target 만들기
                try:
                    err, cur_pose = robot.GetActualTCPPose(flag=1)
                except Exception as e:
                    print(f"\n[ERROR] GetActualTCPPose 예외: {e}\n")
                    continue

                if err != 0:
                    print(f"\n[FAIL] GetActualTCPPose err={err}, pose={cur_pose}\n")
                    continue

                step_pose = blend_pose(cur_pose, last_target_pose, STEP_SCALE)

                print("\n[ACTION] MoveL 실행 ✅")
                print("current      :", fmt_pose6(cur_pose))
                print("target_pose  :", fmt_pose6(last_target_pose))
                print(f"step({STEP_SCALE}) :", fmt_pose6(step_pose))

                try:
                    rtn = robot.MoveL(
                        desc_pos=ensure_pose6(step_pose),
                        tool=tool,
                        user=user,
                        vel=vel,
                        acc=acc,
                        ovl=ovl,
                        blendR=blendR
                    )
                    print(f"[RET] MoveL errcode: {rtn}\n")
                except Exception as e:
                    print(f"[ERROR] MoveL 예외: {e}\n")

            # 7) MoveCart 이동
            elif cmd == "7":
                if last_target_pose is None:
                    print("\n[WARN] 3번으로 target_pose 먼저 생성.\n")
                    continue

                try:
                    err, cur_pose = robot.GetActualTCPPose(flag=1)
                except Exception as e:
                    print(f"\n[ERROR] GetActualTCPPose 예외: {e}\n")
                    continue

                if err != 0:
                    print(f"\n[FAIL] GetActualTCPPose err={err}, pose={cur_pose}\n")
                    continue

                step_pose = blend_pose(cur_pose, last_target_pose, STEP_SCALE)

                print("\n[ACTION] MoveCart 실행 ✅")
                print("current      :", fmt_pose6(cur_pose))
                print("target_pose  :", fmt_pose6(last_target_pose))
                print(f"step({STEP_SCALE}) :", fmt_pose6(step_pose))

                try:
                    rtn = robot.MoveCart(
                        desc_pos=ensure_pose6(step_pose),
                        tool=tool,
                        user=user,
                        vel=vel,
                        acc=acc,
                        ovl=ovl,
                        blendT=blendT
                    )
                    print(f"[RET] MoveCart errcode: {rtn}\n")
                except Exception as e:
                    print(f"[ERROR] MoveCart 예외: {e}\n")

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
