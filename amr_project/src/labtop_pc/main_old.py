# main.py
import sys
import time

LAB_PC_DIR = r"C:\Users\rhdeh\ws_job_msis\amr_project\src\labtop_pc"
if LAB_PC_DIR not in sys.path:
    sys.path.insert(0, LAB_PC_DIR)

from app_config import ROBOT_IP_DEFAULT

from robot import j4_rotate as j4r

# ✅ home_pc 기준 패키지 import (robot/, vision/에 __init__.py 있어야 함)
from robot import robot_config as rc
from robot import connect_fairino as cf
from robot import robot_state as rs
from vision import measure_box as mb
from robot import target_pose as tp
from robot import ik_adjust as ika
from robot import j6_rotate as j6r
from robot import move_step as ms
from robot import return_home as rh
from robot import gripper_control as gc
from robot import smooth_auto as sa
from robot import stack_cycle as sc


def print_menu(robot_connected: bool, robot_ip: str):
    print("\n=======================================")
    print("무엇을 할까요?")
    print("  0 : 로봇 연결/해제 (토글)")
    print("  1 : 현재 tcp_pose + joint(deg) 읽기 (last_tcp_pose 저장, 최초 1회 initial_joint6 저장)")
    print("  2 : 비전 측정 실행 (camXYZ/angle 또는 moveXYZ/angle) (last_measure 저장)")
    print("  3 : target_pose 생성 (1+2 결과 합쳐서 저장)  (ry 보정 포함)")
    print("  4 :  IK 점검/자세 보정 (target_pose)")
    print("  5 :  angle_deg 만큼 J6 회전 (MoveJ)")
    print("  6 :  MoveCart 1-step (phase0: XY+Zhold / phase1: Zdown)")
    print("  7 : 프로그램 시작(초기) 위치로 복귀 (MoveJ initial_joint6)")
    print("  8 :  Gripper Open/Close/Toggle ")
    print("  9 : Smooth Auto (phase0 검증/보정 -> phase0 한번에 -> J6 -> Zdown 한번에) ")
    print(" 10 :  STACK Cycle (A -> DROP(z+48*cnt) -> OPEN -> A -> HOME, cnt++) ")
    print(" 11 :  J4 수동 회전 (deg 입력 -> MoveJ) ")
    print("  q : 종료")
    print("---------------------------------------")
    print(f"  상태: {'CONNECTED ' if robot_connected else 'DISCONNECTED '}")
    print(f"  IP  : {robot_ip}")
    print(f"  VISION: {'RUNNING' if vision_running else 'STOPPED'}")
    print("=======================================")


def _print_measure(res: dict):
    """measure 결과를 사람이 보기 좋게 출력 (camXYZ or moveXYZ 둘 다 지원)"""
    angle = float(res.get("angle_deg", 0.0))

    if "cam_x_mm" in res and "cam_y_mm" in res and "cam_z_mm" in res:
        print(f"camXYZ(mm) = ({float(res['cam_x_mm']):+.1f}, {float(res['cam_y_mm']):+.1f}, {float(res['cam_z_mm']):+.1f})")
    elif "move_x_mm" in res and "move_y_mm" in res and "move_z_mm" in res:
        print(f"moveXYZ(mm)= ({float(res['move_x_mm']):+.1f}, {float(res['move_y_mm']):+.1f}, {float(res['move_z_mm']):+.1f})")
    else:
        # 모르는 형태면 통째로 보여줌
        print("measure_res:", res)

    print(f"angle(deg)  = {angle:+.2f}")


def main():
    robot = None
    robot_ip = ROBOT_IP_DEFAULT

    # ✅ CMD 상태 변수들 (핵심)
    last_tcp_pose = None
    last_measure = None
    last_target_pose = None
    initial_joint6 = None

    # ✅ Gripper 상태
    state = {
        "gripper_activated": False,
        "gripper_closed": None,   # True/False/None
        "stack_counter": 0,
    }

    # (나중 확장 대비)
    approach_phase = 0
    reached_final = False

    def reconnect():
        nonlocal robot
        if robot is not None:
            try:
                cf.disconnect(robot)
            except Exception:
                pass
        time.sleep(0.2)
        robot = cf.connect(robot_ip)

    while True:
        print_menu(robot_connected=(robot is not None), robot_ip=robot_ip)
        cmd = input("입력 (0/1/2/3/4/5/6/7/8/9/10/q) > ").strip().lower()

        if cmd == "0":
            # 토글: 연결돼 있으면 끊고, 아니면 연결
            if robot is not None:
                cf.disconnect(robot)
                robot = None
                continue

            ip_in = input(f"로봇 IP 입력 (엔터=기본 {robot_ip}) > ").strip()
            if ip_in:
                robot_ip = ip_in

            try:
                robot = cf.connect(robot_ip)
            except Exception as e:
                robot = None
                print(f"[FAIL] 연결 실패: {e}")
                time.sleep(0.3)

        elif cmd == "1":
            if robot is None:
                print("[WARN] 로봇이 연결되어 있지 않습니다. (0번으로 먼저 연결)\n")
                continue

            print("\n[ACTION] GetActualTCPPose / Joint...")
            (e1, pose6), (e2, joint6) = rs.read_pose_joint(robot, reconnect=reconnect)
            if e1 != 0 or e2 != 0:
                print(f"[FAIL] err_p={e1}, err_j={e2}\n")
                continue

            last_tcp_pose = pose6

            if initial_joint6 is None:
                initial_joint6 = joint6[:]
                print("[INIT] initial_joint6 저장 ✅ (1번 최초 실행 기준)")
                print("init_joint:", rs.fmt_joint(initial_joint6))

            print("[OK] 현재 상태 ✅ (last_tcp_pose 저장됨)")
            print("tcp_pose :", rs.fmt_pose6(pose6))
            print("joint6   :", rs.fmt_joint(joint6))
            print()

        elif cmd == "2":
            print("\n[ACTION] Vision measure...")
            try:
                res = mb.measure_box()  # dict 또는 None
            except Exception as e:
                print(f"[ERROR] Vision measure 예외: {e}\n")
                res = None

            if res is None:
                print("[FAIL] 측정 실패(None)\n")
                continue

            last_measure = res

            print("[OK] 측정 결과 ✅ (last_measure 저장됨)")
            _print_measure(res)
            if "video_path" in res:
                print(f"video_path = {res['video_path']}")
            print()

        elif cmd == "3":
            if last_tcp_pose is None or last_measure is None:
                print("\n[WARN] 1,2번 먼저 실행해야 합니다. (last_tcp_pose / last_measure 필요)\n")
                continue

            try:
                # ✅ 디버그 포함 버전 우선 사용 (있으면 moveXYZ/angle 같이 출력 가능)
                if hasattr(tp, "build_target_pose_with_debug"):
                    out = tp.build_target_pose_with_debug(last_tcp_pose, last_measure)
                    last_target_pose = out["target_pose6"]

                    print("\n[OK] target_pose 생성/저장 ✅ (orientation 유지)")
                    print("current_pose :", tp.fmt_pose6(last_tcp_pose))
                    print("target_pose  :", tp.fmt_pose6(last_target_pose))

                    print(
                        "moveXYZ(mm)  : "
                        f"({float(out['move_x_mm']):+.1f}, {float(out['move_y_mm']):+.1f}, {float(out['move_z_mm']):+.1f})"
                    )
                    print(f"angle(deg)   : {float(out.get('angle_deg', 0.0)):+.2f}")

                else:
                    # ✅ fallback
                    last_target_pose = tp.build_target_pose(last_tcp_pose, last_measure)

                    print("\n[OK] target_pose 생성/저장 ✅ (orientation 유지)")
                    print("current_pose :", tp.fmt_pose6(last_tcp_pose))
                    print("target_pose  :", tp.fmt_pose6(last_target_pose))

            except Exception as e:
                print(f"\n[ERROR] target_pose 생성 실패: {e}\n")
                continue

            # 이후 7번(이동)에서 쓸 상태 초기화
            approach_phase = 0
            reached_final = False
            print("phase        : 0 (XY+Zhold)")
            print()


        elif cmd == "4":
            if robot is None:
                print("[WARN] 로봇이 연결되어 있지 않습니다. (0번으로 먼저 연결)\n")
                continue
            if last_target_pose is None:
                print("[WARN] target_pose가 없습니다. 3번을 먼저 실행하세요.\n")
                continue

            (e1, cur_pose6), (e2, cur_joint6) = rs.read_pose_joint(robot, reconnect=reconnect)
            if e1 != 0 or e2 != 0:
                print(f"[FAIL] err_p={e1}, err_j={e2}\n")
                continue

            print("\n[ACTION] CMD4 IK check + adjust (target + phase0 + step_phase0)...")

            out = ika.cmd4_check_and_adjust_target_only(
                robot=robot,
                reconnect=reconnect,
                cur_pose6=cur_pose6,
                cur_joint6=cur_joint6,
                target_pose6=last_target_pose,
            )

            if not out.get("ok", False):
                print("[CMD4] ❌ 실패")
                print("  msg   :", out.get("msg"))
                if out.get("flags") is not None:
                    print("  flags :", out.get("flags"))
                if out.get("target") is not None:
                    print("  target:", ika.fmt_pose6(out.get("target")))
                if out.get("phase0") is not None:
                    print("  phase0:", ika.fmt_pose6(out.get("phase0")))
                if out.get("step_phase0") is not None:
                    print("  step0 :", ika.fmt_pose6(out.get("step_phase0")))
                print()
                continue

            last_target_pose = out["target"]

            if out.get("adjusted", False):
                drx, dry, drz = out["d"]
                print("[CMD4] ✅ 보정 성공! last_target_pose 업데이트")
                print(f"  tried={out.get('tries')} score={out.get('score'):.3f}")
                print(f"  dRPY(deg)=({drx:+.1f}, {dry:+.1f}, {drz:+.1f})")
            else:
                print("[CMD4] ✅ 이미 IK OK (target+phase0+step0 모두 OK)")

            print("  target:", ika.fmt_pose6(out["target"]))
            if out.get("phase0") is not None:
                print("  phase0:", ika.fmt_pose6(out["phase0"]))
            if out.get("step_phase0") is not None:
                print("  step0 :", ika.fmt_pose6(out["step_phase0"]))
            print()

        elif cmd == "5":
            if robot is None:
                print("[WARN] 로봇이 연결되어 있지 않습니다. (0번으로 먼저 연결)\n")
                continue
            if last_measure is None:
                print("\n[WARN] 2번 먼저 실행해야 angle 값이 있어요.\n")
                continue

            ok, delta, err = j6r.rotate_j6_from_measure(
                robot=robot,
                last_measure=last_measure,
                reconnect=reconnect
            )

            if not ok:
                print(f"[FAIL] J6 rotate failed err={err}\n")
            else:
                print(f"[OK] J6 rotate done delta={delta:+.3f} deg\n")

        elif cmd == "6":
            if robot is None:
                print("[WARN] 로봇이 연결되어 있지 않습니다. (0번으로 먼저 연결)\n")
                continue
            if last_target_pose is None:
                print("[WARN] target_pose가 없습니다. 3번을 먼저 실행하세요.\n")
                continue

            out = ms.cmd7_run(
                robot=robot,
                reconnect=reconnect,
                last_target_pose6=last_target_pose,
                approach_phase=approach_phase,
                reached_final=reached_final,
                step_scale=rc.STEP_SCALE_DEFAULT
            )

            if not out["ok"]:
                print(f"[CMD7-step] ❌ {out['msg']}\n")
                continue

            approach_phase = int(out["new_phase"])
            reached_final = bool(out["reached_final"])

            print(f"\n[CMD7-step] ✅ {out['msg']}")
            if out["pose_after"] is not None and out["joint_after"] is not None:
                print("  new_pose  :", rs.fmt_pose6(out["pose_after"]))
                print("  new_joint :", rs.fmt_joint(out["joint_after"]))
            print(f"  phase     : {approach_phase}  (0=XY+Zhold, 1=Zdown)")
            print(f"  reached   : {reached_final}")
            dbg = out.get("debug", {})
            if dbg:
                print(f"  used_st   : {dbg.get('used_st')}  xyz_scale=({dbg.get('sx')},{dbg.get('sy')},{dbg.get('sz')})  ori_s={dbg.get('ori_s')}")
            print()

        elif cmd == "7":
            if robot is None:
                print("\n[WARN] 로봇 연결 먼저(0번)\n")
                continue
            if initial_joint6 is None:
                print("\n[WARN] initial_joint6가 없습니다. 1번을 먼저 눌러 초기 joint를 저장하세요.\n")
                continue

            out = rh.cmd7_return_to_initial(
                robot=robot,
                initial_joint6=initial_joint6,
                reconnect=reconnect
            )

            if not out["ok"]:
                print(f"\n[FAIL] {out['msg']}\n")
                continue

            approach_phase = 0
            reached_final = False
            print("\n[RESET] phase=0, reached_final=False 로 초기화 ✅\n")

        elif cmd == "8":
            if robot is None:
                print("[WARN] 로봇이 연결되어 있지 않습니다. (0번으로 먼저 연결)\n")
                continue

            gc.run_gripper_menu(
                robot=robot,
                reconnect=reconnect,
                state=state
            )

        elif cmd == "9":
            if robot is None:
                print("[WARN] 로봇이 연결되어 있지 않습니다. (0번으로 먼저 연결)\n")
                continue
            if last_target_pose is None:
                print("[WARN] target_pose가 없습니다. 3번/4번을 먼저 실행하세요.\n")
                continue
            if last_measure is None:
                print("[WARN] last_measure가 없습니다. 2번(비전 측정)을 먼저 실행하세요.\n")
                continue

            out = sa.cmd9_smooth_auto(
                robot=robot,
                reconnect=reconnect,
                last_target_pose6=last_target_pose,
                last_measure=last_measure,
                state=state,
                tool=0,
                user=0,
                auto_grip_close=True,
            )

            if not out["ok"]:
                print(f"[CMD9] ❌ {out['msg']}\n")
                if out.get("phase0") is not None:
                    print("  phase0:", rs.fmt_pose6(out["phase0"]))
                if out.get("zdown") is not None:
                    print("  zdown :", rs.fmt_pose6(out["zdown"]))
                print()
                continue

            reached_final = True
            approach_phase = 0
            print(f"[CMD9] ✅ {out['msg']}\n")

        elif cmd == "10":
            if robot is None:
                print("[WARN] 로봇이 연결되어 있지 않습니다. (0번으로 먼저 연결)\n")
                continue
            if initial_joint6 is None:
                print("[WARN] initial_joint6가 없습니다. 1번을 먼저 눌러 초기 joint를 저장하세요.\n")
                continue

            out = sc.cmd11_stack_cycle(
                robot=robot,
                reconnect=reconnect,
                state=state,
                home_joint6=initial_joint6,
                tool=0,
                user=0
            )

            if not out["ok"]:
                print(f"[CMD11] ❌ {out['msg']}\n")
            else:
                print(f"[CMD11] ✅ {out['msg']}\n")
                
        elif cmd == "11":  # ✅ J4 입력 회전
            if robot is None:
                print("[ERR] 로봇 연결부터 해 (0번)")
                continue

            s = input("J4 delta(deg) 입력 (예: 5, -10): ").strip()
            try:
                delta = float(s)
            except Exception:
                print("[ERR] 숫자로 입력해")
                continue

            ok, d, err = j4r.rotate_j4_from_input(
                robot=robot,
                delta_deg=delta,
                tool=0,
                user=0,
                reconnect=reconnect,
            )
            print(f"[DONE] ok={ok}, delta={d:+.3f}, err={err}\n")


        elif cmd == "q":
            print("[EXIT] 종료합니다.")
            break

        else:
            print("[WARN] 잘못된 입력입니다.")

    if robot is not None:
        cf.disconnect(robot)


if __name__ == "__main__":
    main()

