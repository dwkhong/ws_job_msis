import sys
import time

# ✅ 1번에서 실행할 측정 코드: box_test_3.py의 main()이 dict 또는 None 반환해야 함
from box_test_3 import main as measure_main

# ✅ Robot.cp39-win_amd64.pyd 폴더를 Python 경로에 추가
sys.path.insert(
    0,
    r"C:\Users\rhdeh\ws_job_msis\amr_project\driver\fairino-python-sdk-main\windows\fairino\build\lib.win-amd64-cpython-39"
)

# ✅ "Robot 모듈(.pyd)" 직접 import
import Robot


# -------------------------
# Config
# -------------------------
STEP_SCALE = 0.1  # ✅ 10분의 1씩만 움직이기 (0.1 = 1/10)


# -------------------------
# Utils
# -------------------------
def fmt_pose6(pose):
    if not isinstance(pose, (list, tuple)) or len(pose) < 6:
        return str(pose)
    x, y, z, rx, ry, rz = pose[:6]
    return f"[x,y,z,rx,ry,rz]=[{x:.3f}, {y:.3f}, {z:.3f}, {rx:.3f}, {ry:.3f}, {rz:.3f}]"


def ensure_pose6(p):
    """MoveL/MoveCart용: pose는 항상 6개 [x,y,z,rx,ry,rz]로 강제."""
    if not isinstance(p, (list, tuple)):
        raise ValueError(f"pose is not list/tuple: {type(p)}")
    if len(p) < 6:
        raise ValueError(f"pose length < 6: {len(p)}")
    return list(p[:6])


def build_target_pose(cur_pose, move_res, scale=1.0):
    """
    ✅ 각도(angle)는 적용하지 않고(저장만), XYZ만 반영해서 target_pose 생성
    scale=0.1이면 moveXYZ를 1/10만 적용해서 target_pose 생성
    cur_pose: [x,y,z,rx,ry,rz] (mm, deg)
    move_res: dict from box_test_3 main()
      - move_x_mm, move_y_mm, move_z_mm, angle_deg (angle은 사용하지 않음)
    return: target_pose(list[6])
    """
    x, y, z, rx, ry, rz = ensure_pose6(cur_pose)

    dx = float(move_res["move_x_mm"]) * scale
    dy = float(move_res["move_y_mm"]) * scale
    dz = float(move_res["move_z_mm"]) * scale

    tx = x + dx
    ty = y + dy
    tz = z + dz

    # ✅ 자세는 그대로 유지 (angle 미적용)
    return [tx, ty, tz, rx, ry, rz]


def prompt_menu():
    print("=======================================")
    print("무슨 기능을 할까요?")
    print("  1 : box_test_3 측정 실행 -> moveXYZ/angle 저장")
    print("  2 : GetActualTCPPose(flag=1) -> 현재 tcp_pose 저장 (원본 유지)")
    print("  3 : (1)+(2) 합쳐서 target_pose 생성(저장/출력)  ※ angle은 적용 안함")
    print(f"  4 : MoveL(현재Pose + moveXYZ*{STEP_SCALE})  ✅ 직선 이동 (LIN)")
    print(f"  5 : MoveJ(현재Pose + moveXYZ*{STEP_SCALE})  ✅ 관절 이동 (PTP)")
    print(f"  6 : MoveCart(현재Pose + moveXYZ*{STEP_SCALE}) ✅ 카테시안 PTP")
    print("  q : 종료")
    print("=======================================")
    return input("입력 (1/2/3/4/5/6/q) > ").strip()


def main():
    print("[INFO] Connecting to Fairino...")
    robot = Robot.RPC("192.168.58.3")
    print("[INFO] Robot connected ✅\n")

    # -------------------------
    # Motion params (원하면 바꿔)
    # -------------------------
    tool = 0
    user = 0
    vel = 20.0     # 0~100 (%)
    acc = 0.0      # 문서상 미오픈인 경우 많음
    ovl = 100.0
    blendT = -1.0  # blocking
    blendR = -1.0  # blocking

    last_measure = None       # 1번 결과(dict)
    last_tcp_pose = None      # 2번 결과(pose) - 원본
    last_target_pose = None   # 3번 결과([x,y,z,rx,ry,rz]) - 원본 moveXYZ(100%) 기준

    try:
        while True:
            cmd = prompt_menu()

            # -------------------------
            # 1) 측정 실행
            # -------------------------
            if cmd == "1":
                print("\n[ACTION] box_test_3 측정 시작...")
                res = measure_main()

                if res is None:
                    print("[FAIL] 측정 실패(타임아웃/유효샘플 부족)\n")
                    continue

                last_measure = res
                mx = float(res["move_x_mm"])
                my = float(res["move_y_mm"])
                mz = float(res["move_z_mm"])
                ang = float(res.get("angle_deg", 0.0))

                print("\n[OK] 측정 결과 저장 ✅")
                print(f"moveXYZ(mm) = ({mx:+.1f}, {my:+.1f}, {mz:+.1f})")
                print(f"angle(deg)  = {ang:+.2f}  (saved only, NOT applied)\n")

            # -------------------------
            # 2) 현재 TCP pose 읽기 (원본 유지)
            # -------------------------
            elif cmd == "2":
                print("\n[ACTION] GetActualTCPPose(flag=1) 호출...")
                try:
                    err, tcp_pose = robot.GetActualTCPPose(flag=1)
                except Exception as e:
                    print(f"[ERROR] GetActualTCPPose 호출 예외: {e}\n")
                    continue

                if err != 0:
                    print(f"[FAIL] GetActualTCPPose 실패 err={err}, tcp_pose={tcp_pose}\n")
                    continue

                last_tcp_pose = tcp_pose  # 원본 저장
                print("[OK] 현재 tcp_pose 저장 ✅ (원본)")
                print(fmt_pose6(last_tcp_pose))
                print()

            # -------------------------
            # 3) target_pose 생성 (1 + 2) - 원본(100%) 이동량 기준으로 계산/저장
            # -------------------------
            elif cmd == "3":
                if last_measure is None:
                    print("\n[WARN] 1번(측정)을 먼저 실행해야 함.\n")
                    continue
                if last_tcp_pose is None:
                    print("\n[WARN] 2번(TCP pose)을 먼저 실행해야 함.\n")
                    continue

                last_target_pose = build_target_pose(last_tcp_pose, last_measure, scale=1.0)

                print("\n[OK] target_pose 생성/저장 ✅ (원본 moveXYZ 100% 기준)")
                print("current_pose :", fmt_pose6(last_tcp_pose))
                print(f"moveXYZ(mm)  : ({float(last_measure['move_x_mm']):+.1f}, {float(last_measure['move_y_mm']):+.1f}, {float(last_measure['move_z_mm']):+.1f})")
                print(f"angle(deg)   : {float(last_measure.get('angle_deg', 0.0)):+.2f}  (NOT applied)")
                print("target_pose  :", fmt_pose6(last_target_pose))
                print()

            # -------------------------
            # 4) MoveL: Cartesian Linear (실행 시 현재 pose 기준 + 1/10 step)
            # -------------------------
            elif cmd == "4":
                if last_measure is None:
                    print("\n[WARN] 1번(측정)을 먼저 실행해야 함.\n")
                    continue

                try:
                    err, cur_pose = robot.GetActualTCPPose(flag=1)
                except Exception as e:
                    print(f"\n[ERROR] GetActualTCPPose 예외: {e}\n")
                    continue

                if err != 0:
                    print(f"\n[FAIL] GetActualTCPPose 실패 err={err}, pose={cur_pose}\n")
                    continue

                desc_pos = build_target_pose(cur_pose, last_measure, scale=STEP_SCALE)

                print("\n[ACTION] MoveL 실행 (LIN) - 1/10 step ✅")
                print("current :", fmt_pose6(cur_pose))
                print(f"stepXYZ(mm): ({float(last_measure['move_x_mm'])*STEP_SCALE:+.1f}, {float(last_measure['move_y_mm'])*STEP_SCALE:+.1f}, {float(last_measure['move_z_mm'])*STEP_SCALE:+.1f})")
                print("target  :", fmt_pose6(desc_pos))

                try:
                    rtn = robot.MoveL(
                        desc_pos=ensure_pose6(desc_pos),
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

            # -------------------------
            # 5) MoveJ: Joint motion (실행 시 현재 pose 기준 + 1/10 step)
            # -------------------------
            elif cmd == "5":
                if last_measure is None:
                    print("\n[WARN] 1번(측정)을 먼저 실행해야 함.\n")
                    continue

                try:
                    err, cur_pose = robot.GetActualTCPPose(flag=1)
                except Exception as e:
                    print(f"\n[ERROR] GetActualTCPPose 예외: {e}\n")
                    continue

                if err != 0:
                    print(f"\n[FAIL] GetActualTCPPose 실패 err={err}, pose={cur_pose}\n")
                    continue

                desc_pos = build_target_pose(cur_pose, last_measure, scale=STEP_SCALE)

                # 현재 관절각 얻기 (flag=1)
                try:
                    ret, j = robot.GetActualJointPosDegree(1)
                except Exception as e:
                    print(f"\n[ERROR] GetActualJointPosDegree 예외: {e}\n")
                    continue

                if ret != 0:
                    print(f"\n[FAIL] GetActualJointPosDegree 실패 ret={ret}, j={j}\n")
                    continue

                joint_pos = list(j[:6])

                print("\n[ACTION] MoveJ 실행 (PTP) - 1/10 step ✅")
                print("current :", fmt_pose6(cur_pose))
                print("target  :", fmt_pose6(desc_pos))
                print("joint_pos(current):", [f"{v:.3f}" for v in joint_pos])

                try:
                    rtn = robot.MoveJ(
                        joint_pos=joint_pos,
                        tool=tool,
                        user=user,
                        desc_pos=ensure_pose6(desc_pos),  # ✅ IK 힌트/검증용
                        vel=vel,
                        acc=acc,
                        ovl=ovl,
                        blendT=blendT
                    )
                    print(f"[RET] MoveJ errcode: {rtn}\n")
                except Exception as e:
                    print(f"[ERROR] MoveJ 예외: {e}\n")

            # -------------------------
            # 6) MoveCart: Cartesian PTP (실행 시 현재 pose 기준 + 1/10 step)
            # -------------------------
            elif cmd == "6":
                if last_measure is None:
                    print("\n[WARN] 1번(측정)을 먼저 실행해야 함.\n")
                    continue

                try:
                    err, cur_pose = robot.GetActualTCPPose(flag=1)
                except Exception as e:
                    print(f"\n[ERROR] GetActualTCPPose 예외: {e}\n")
                    continue

                if err != 0:
                    print(f"\n[FAIL] GetActualTCPPose 실패 err={err}, pose={cur_pose}\n")
                    continue

                desc_pos = build_target_pose(cur_pose, last_measure, scale=STEP_SCALE)

                print("\n[ACTION] MoveCart 실행 (Cartesian PTP) - 1/10 step ✅")
                print("current :", fmt_pose6(cur_pose))
                print("target  :", fmt_pose6(desc_pos))

                try:
                    rtn = robot.MoveCart(
                        desc_pos=ensure_pose6(desc_pos),
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

            # -------------------------
            # 종료
            # -------------------------
            elif cmd.lower() == "q":
                print("\n[EXIT] 종료합니다.")
                break

            else:
                print("\n[WARN] 잘못된 입력입니다. 1/2/3/4/5/6/q 중에서 선택하세요.\n")

            time.sleep(0.1)

    finally:
        # 혹시라도 예외로 튕겨도 연결 종료
        try:
            robot.CloseRPC()
        except Exception:
            pass


if __name__ == "__main__":
    main()
