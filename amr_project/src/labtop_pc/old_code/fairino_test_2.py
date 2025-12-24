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


def fmt_pose6(pose):
    if not isinstance(pose, (list, tuple)) or len(pose) < 6:
        return str(pose)
    x, y, z, rx, ry, rz = pose[:6]
    return f"[x,y,z,rx,ry,rz]=[{x:.3f}, {y:.3f}, {z:.3f}, {rx:.3f}, {ry:.3f}, {rz:.3f}]"


def build_target_pose(cur_pose, move_res):
    """
    ✅ 각도(angle)는 적용하지 않고(저장만), XYZ만 반영해서 target_pose 생성
    cur_pose: [x,y,z,rx,ry,rz] (mm, deg)
    move_res: dict from box_test_3 main()
      - move_x_mm, move_y_mm, move_z_mm, angle_deg (angle은 사용하지 않음)
    return: target_pose(list[6])
    """
    x, y, z, rx, ry, rz = cur_pose[:6]

    dx = float(move_res["move_x_mm"])
    dy = float(move_res["move_y_mm"])
    dz = float(move_res["move_z_mm"])

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
    print("  q : 종료")
    print("=======================================")
    return input("입력 (1/2/3/q) > ").strip()


def main():
    print("[INFO] Connecting to Fairino...")
    robot = Robot.RPC("192.168.58.3")
    print("[INFO] Robot connected ✅\n")

    last_measure = None       # ✅ 1번 결과(dict)
    last_tcp_pose = None      # ✅ 2번 결과([x,y,z,rx,ry,rz]) - 원본 그대로 유지
    last_target_pose = None   # ✅ 3번에서 만든 target_pose([x,y,z,rx,ry,rz])

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
            mx = res["move_x_mm"]
            my = res["move_y_mm"]
            mz = res["move_z_mm"]
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

            # ✅ 원본 그대로 저장 (수정/가공 X)
            last_tcp_pose = tcp_pose

            print("[OK] 현재 tcp_pose 저장 ✅ (원본)")
            print(fmt_pose6(last_tcp_pose))
            print()

        # -------------------------
        # 3) target_pose 생성 (1 + 2)  ※ angle 미적용
        # -------------------------
        elif cmd == "3":
            if last_measure is None:
                print("\n[WARN] 1번(측정)을 먼저 실행해야 함.\n")
                continue
            if last_tcp_pose is None:
                print("\n[WARN] 2번(TCP pose)을 먼저 실행해야 함.\n")
                continue

            last_target_pose = build_target_pose(last_tcp_pose, last_measure)

            print("\n[OK] target_pose 생성/저장 ✅")
            print("current_pose :", fmt_pose6(last_tcp_pose))
            print(f"moveXYZ(mm)  : ({last_measure['move_x_mm']:+.1f}, {last_measure['move_y_mm']:+.1f}, {last_measure['move_z_mm']:+.1f})")
            print(f"angle(deg)   : {float(last_measure.get('angle_deg', 0.0)):+.2f}  (NOT applied)")
            print("target_pose  :", fmt_pose6(last_target_pose))
            print()

        # -------------------------
        # 종료
        # -------------------------
        elif cmd.lower() == "q":
            print("\n[EXIT] 종료합니다.")
            break

        else:
            print("\n[WARN] 잘못된 입력입니다. 1/2/3/q 중에서 선택하세요.\n")

        # 보기 좋게 살짝 텀
        time.sleep(0.1)


if __name__ == "__main__":
    main()







