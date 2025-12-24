import sys
import time

from box_test_3 import main as measure_main

# ✅ Robot.cp39-win_amd64.pyd 폴더를 Python 경로에 추가
sys.path.insert(
    0,
    r"C:\Users\rhdeh\ws_job_msis\amr_project\driver\fairino-python-sdk-main\windows\fairino\build\lib.win-amd64-cpython-39"
)
import Robot

ROBOT_IP = "192.168.58.3"
STEP_SCALE = 0.1  # 0.1 = 1/10 step


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
    return "[" + ", ".join(f"{float(v):.3f}" for v in j) + "]"


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


def ensure_joint7(j):
    """
    ✅ MoveL 내부 포맷이 desc(6)+joint(7)=13을 기대하는 케이스 대응
    - 6개면 7번째(외부축/예약축) 0.0을 붙인다.
    """
    if not isinstance(j, (list, tuple)):
        raise ValueError(f"joint is not list/tuple: {type(j)}")
    if len(j) >= 7:
        return [float(x) for x in j[:7]]
    if len(j) == 6:
        return [float(x) for x in list(j) + [0.0]]
    raise ValueError(f"joint length invalid: {len(j)}")


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


def get_actual_joint_deg(robot, flag=1):
    # ✅ 문서 함수명 그대로
    err, joint_pos = robot.GetActualJointPosDegree(flag=flag)
    return err, joint_pos


def prompt_menu():
    print("=======================================")
    print("무슨 기능을 할까요?")
    print("  1 : 현재 tcp_pose + joint(deg) 가져오기/저장")
    print("  2 : box_test_3 측정 실행/저장 (moveXYZ/angle)")
    print("  3 : target_pose 생성/저장 (1+2)")
    print("  4 : IK 가능/불가능 확인 (HasSolution)  ※ target_pose 기준")
    print("  5 : IK 해(joint 솔루션) 계산해서 출력  ※ target_pose 기준")
    print(f"  6 : MoveL 이동 (현재→target * {STEP_SCALE})  ✅ joint_pos=7 강제 + 위치인자 호출")
    print(f"  7 : MoveCart 이동 (현재→target * {STEP_SCALE})")
    print("  q : 종료")
    print("=======================================")
    return input("입력 (1/2/3/4/5/6/7/q) > ").strip()


def main():
    print("[INFO] Connecting to Fairino...")
    robot = Robot.RPC(ROBOT_IP)
    print("[INFO] Robot connected ✅\n")

    # -------------------------
    # Motion params
    # -------------------------
    tool = 0
    user = 0
    vel = 20.0
    acc = 0.0
    ovl = 100.0
    blendT = -1.0
    blendR = -1.0
    config = -1

    # MoveL 문서 기본 파라미터들
    blendMode = 0
    exaxis_pos = [0.0, 0.0, 0.0, 0.0]
    search = 0
    offset_flag = 0
    offset_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    velAccParamMode = 0
    overSpeedStrategy = 0
    speedPercent = 10

    # 저장값
    last_tcp_pose = None
    last_joint6 = None
    last_measure = None
    last_target_pose = None

    try:
        while True:
            cmd = prompt_menu()

            # 1) 현재 tcp + joint 저장
            if cmd == "1":
                print("\n[ACTION] GetActualTCPPose(flag=1)...")
                try:
                    err_p, tcp_pose = robot.GetActualTCPPose(flag=1)
                except Exception as e:
                    print(f"[ERROR] GetActualTCPPose 예외: {e}\n")
                    continue
                if err_p != 0:
                    print(f"[FAIL] GetActualTCPPose err={err_p}, pose={tcp_pose}\n")
                    continue

                print("[ACTION] GetActualJointPosDegree(flag=1)...")
                try:
                    err_j, joint_pos = get_actual_joint_deg(robot, flag=1)
                except Exception as e:
                    print(f"[ERROR] GetActualJointPosDegree 예외: {e}\n")
                    continue
                if err_j != 0:
                    print(f"[FAIL] GetActualJointPosDegree err={err_j}, joint={joint_pos}\n")
                    continue

                last_tcp_pose = ensure_pose6(tcp_pose)
                last_joint6 = ensure_joint6(joint_pos)

                print("[OK] 현재 상태 저장 ✅")
                print("tcp_pose :", fmt_pose6(last_tcp_pose))
                print("joint6   :", fmt_joint(last_joint6))
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

            # 4) IK 가능/불가능 (target_pose 기준)
            elif cmd == "4":
                if last_target_pose is None:
                    print("\n[WARN] 3번으로 target_pose 먼저 생성.\n")
                    continue

                try:
                    err_j, cur_joint = get_actual_joint_deg(robot, flag=1)
                except Exception as e:
                    print(f"[ERROR] GetActualJointPosDegree 예외: {e}\n")
                    continue
                if err_j != 0:
                    print(f"[FAIL] GetActualJointPosDegree err={err_j}, joint={cur_joint}\n")
                    continue

                cur_joint6 = ensure_joint6(cur_joint)
                target = ensure_pose6(last_target_pose)

                try:
                    err, result = robot.GetInverseKinHasSolution(0, target, cur_joint6)
                except Exception as e:
                    print(f"[ERROR] GetInverseKinHasSolution 예외: {e}\n")
                    continue

                if err != 0:
                    print(f"[FAIL] HasSolution err={err}, result={result}\n")
                else:
                    print(f"[RESULT] HasSolution(target_pose) = {bool(result)}\n")

            # 5) IK 해 출력 (target_pose 기준)
            elif cmd == "5":
                if last_target_pose is None:
                    print("\n[WARN] 3번으로 target_pose 먼저 생성.\n")
                    continue

                try:
                    err_j, cur_joint = get_actual_joint_deg(robot, flag=1)
                except Exception as e:
                    print(f"[ERROR] GetActualJointPosDegree 예외: {e}\n")
                    continue
                if err_j != 0:
                    print(f"[FAIL] GetActualJointPosDegree err={err_j}, joint={cur_joint}\n")
                    continue

                cur_joint6 = ensure_joint6(cur_joint)
                target = ensure_pose6(last_target_pose)

                try:
                    err_sol, has_sol = robot.GetInverseKinHasSolution(0, target, cur_joint6)
                except Exception as e:
                    print(f"[ERROR] GetInverseKinHasSolution 예외: {e}\n")
                    continue
                if err_sol != 0:
                    print(f"[FAIL] HasSolution err={err_sol}, result={has_sol}\n")
                    continue
                if not bool(has_sol):
                    print("[RESULT] ❌ IK 해 없음: target 도달 불가\n")
                    continue

                try:
                    err_ik, joint_sol = robot.GetInverseKinRef(0, target, cur_joint6)
                except Exception as e:
                    print(f"[ERROR] GetInverseKinRef 예외: {e}\n")
                    continue
                if err_ik != 0:
                    print(f"[FAIL] GetInverseKinRef err={err_ik}, joint_sol={joint_sol}\n")
                    continue

                joint_sol6 = ensure_joint6(joint_sol)
                print("[OK] IK Joint Solution ✅")
                print("current_joint6:", fmt_joint(cur_joint6))
                print("joint_sol6    :", fmt_joint(joint_sol6))
                print()

            # 6) MoveL
            elif cmd == "6":
                if last_target_pose is None:
                    print("\n[WARN] 3번으로 target_pose 먼저 생성.\n")
                    continue

                # 현재 TCP
                try:
                    err_p, cur_pose = robot.GetActualTCPPose(flag=1)
                except Exception as e:
                    print(f"\n[ERROR] GetActualTCPPose 예외: {e}\n")
                    continue
                if err_p != 0:
                    print(f"\n[FAIL] GetActualTCPPose err={err_p}, pose={cur_pose}\n")
                    continue

                # 현재 joint
                try:
                    err_j, cur_joint = get_actual_joint_deg(robot, flag=1)
                except Exception as e:
                    print(f"[ERROR] GetActualJointPosDegree 예외: {e}\n")
                    continue
                if err_j != 0:
                    print(f"[FAIL] GetActualJointPosDegree err={err_j}, joint={cur_joint}\n")
                    continue
                cur_joint6 = ensure_joint6(cur_joint)

                # step pose
                step_pose = ensure_pose6(blend_pose(cur_pose, last_target_pose, STEP_SCALE))

                # IK 해 계산
                try:
                    err_sol, has_sol = robot.GetInverseKinHasSolution(0, step_pose, cur_joint6)
                except Exception as e:
                    print(f"[ERROR] GetInverseKinHasSolution 예외: {e}\n")
                    continue
                if err_sol != 0 or not bool(has_sol):
                    print(f"[RESULT] ❌ step_pose IK 해 없음 (err={err_sol}, has={has_sol})\n")
                    continue

                try:
                    err_ik, joint_sol = robot.GetInverseKinRef(0, step_pose, cur_joint6)
                except Exception as e:
                    print(f"[ERROR] GetInverseKinRef 예외: {e}\n")
                    continue
                if err_ik != 0:
                    print(f"[FAIL] GetInverseKinRef err={err_ik}, joint_sol={joint_sol}\n")
                    continue

                joint_sol6 = ensure_joint6(joint_sol)
                joint_sol7 = ensure_joint7(joint_sol6)  # ✅ 7개 강제

                print("\n[ACTION] MoveL 실행 ✅ (positional call + joint_pos 7개)")
                print("LEN step_pose :", len(step_pose), step_pose)
                print("LEN joint_sol6:", len(joint_sol6), joint_sol6)
                print("LEN joint_sol7:", len(joint_sol7), joint_sol7)
                print("current_pose  :", fmt_pose6(cur_pose))
                print("target_pose   :", fmt_pose6(last_target_pose))
                print(f"step({STEP_SCALE})    :", fmt_pose6(step_pose))
                print("config        :", config)

                # ✅ 핵심: keyword 대신 "위치인자"로 문서 순서대로 넣어줌 (래퍼/format 꼬임 방지)
                try:
                    rtn = robot.MoveL(
                        step_pose,        # desc_pos
                        tool,             # tool
                        user,             # user
                        joint_sol7,        # joint_pos (7)
                        vel,              # vel
                        acc,              # acc
                        ovl,              # ovl
                        blendR,           # blendR
                        blendMode,        # blendMode
                        exaxis_pos,       # exaxis_pos (4)
                        search,           # search
                        offset_flag,      # offset_flag
                        offset_pos,       # offset_pos (6)
                        config,           # config
                        velAccParamMode,  # velAccParamMode
                        overSpeedStrategy,# overSpeedStrategy
                        speedPercent      # speedPercent
                    )
                    print(f"[RET] MoveL errcode: {rtn}\n")
                except Exception as e:
                    print(f"[ERROR] MoveL 예외: {e}\n")

            # 7) MoveCart (문서 그대로)
            elif cmd == "7":
                if last_target_pose is None:
                    print("\n[WARN] 3번으로 target_pose 먼저 생성.\n")
                    continue

                try:
                    err_p, cur_pose = robot.GetActualTCPPose(flag=1)
                except Exception as e:
                    print(f"\n[ERROR] GetActualTCPPose 예외: {e}\n")
                    continue
                if err_p != 0:
                    print(f"\n[FAIL] GetActualTCPPose err={err_p}, pose={cur_pose}\n")
                    continue

                step_pose = ensure_pose6(blend_pose(cur_pose, last_target_pose, STEP_SCALE))

                print("\n[ACTION] MoveCart 실행 ✅")
                print("current_pose  :", fmt_pose6(cur_pose))
                print("target_pose   :", fmt_pose6(last_target_pose))
                print(f"step({STEP_SCALE})    :", fmt_pose6(step_pose))

                try:
                    rtn = robot.MoveCart(step_pose, tool, user, vel, acc, ovl, blendT, -1)
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

