# main.py
import sys
import time

LAB_PC_DIR = r"C:\Users\rhdeh\ws_job_msis\amr_project\src\labtop_pc"
if LAB_PC_DIR not in sys.path:
    sys.path.insert(0, LAB_PC_DIR)

from app_config import ROBOT_IP_DEFAULT

from robot import j4_rotate as j4r

# ✅ home_pc 기준 패키지 import
from robot import robot_config as rc
from robot import connect_fairino as cf
from robot import robot_state as rs
from vision import measure_box_2 as mb
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
    print("  0 : 로봇 연결/해제 (토글)  (+ 연결되면 비전 스트리밍 ON)")
    print("  1 : 현재 tcp_pose + joint(deg) 읽기 (last_tcp_pose 저장, 최초 1회 initial_joint6 저장)")
    print("  2 : 비전 측정 요청 (스트리밍 유지 + 프레임 10개 평균) (last_measure 저장)")
    print("  3 : target_pose 생성 (1+2 결과 합쳐서 저장)  (ry 보정 포함)")
    print("  4 :  IK 점검/자세 보정 (target_pose)")
    print("  5 :  angle_deg 만큼 J6 회전 (MoveJ)")
    print("  6 :  MoveCart 1-step (phase0: XY+Zhold / phase1: Zdown)")
    print("  7 : 프로그램 시작(초기) 위치로 복귀 (MoveJ initial_joint6)")
    print("  8 :  Gripper Open/Close/Toggle ")
    print("  9 : Smooth Auto (phase0 검증/보정 -> phase0 한번에 -> J6 -> Zdown 한번에) ")
    print(" 10 :  STACK Cycle (A -> DROP(z+48*cnt) -> OPEN -> A -> HOME, cnt++) ")
    print(" 11 :  J4 수동 회전 (deg 입력 -> MoveJ) ")
    print(" 12 :  ★ AUTO LOOP ★ (Home->Measure->Pick->Home->Stack Loop) ")
    print("  q : 종료")
    print("---------------------------------------")
    print(f"  상태: {'CONNECTED ' if robot_connected else 'DISCONNECTED '}")
    print(f"  IP  : {robot_ip}")
    print(f"  VISION: {'RUNNING' if mb.is_running() else 'STOPPED'}  (ESC=stop)")
    print("=======================================")


def _print_measure(res: dict):
    # 1. 기본 3D 좌표 (X, Y, Z)
    if "move_x_mm" in res:
        print(f"  [Position] XYZ(mm) = ({res['move_x_mm']:.1f}, {res['move_y_mm']:.1f}, {res['move_z_mm']:.1f})")
    
    # 2. 회전 각도 (RZ: 기존 OBB 회전, RX/RY: 이번에 추가한 3D 기울기)
    rz = float(res.get("angle_deg", 0.0))
    ry = float(res.get("ry_deg", 0.0)) # Pitch (앞뒤 기울기)
    rx = float(res.get("rx_deg", 0.0)) # Roll (좌우 기울기)
    
    print(f"  [Rotation] RZ (Yaw)  = {rz:+.2f} deg  (박스 회전)")
    print(f"             RY (Pitch)= {ry:+.2f} deg  (앞뒤 경사) -> 로봇 Pitch 보정용")
    print(f"             RX (Roll) = {rx:+.2f} deg  (좌우 경사)")

    # (참고) 디버깅용 캠 좌표
    if "cam_x_mm" in res:
        print(f"  (Debug-Cam) XYZ    = ({float(res['cam_x_mm']):.1f}, {float(res['cam_y_mm']):.1f}, {float(res['cam_z_mm']):.1f})")


def main():
    robot = None
    robot_ip = ROBOT_IP_DEFAULT

    last_tcp_pose = None
    last_measure = None
    last_target_pose = None
    initial_joint6 = None

    state = {
        "gripper_activated": False,
        "gripper_closed": None,
        "stack_counter": 0,
    }

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
        cmd = input("입력 (0~12/q) > ").strip().lower()

        if cmd == "0":
            # 토글: 연결돼 있으면 끊고, 아니면 연결
            if robot is not None:
                # ✅ 비전 스트리밍도 같이 OFF
                try:
                    mb.stop_stream()
                except Exception:
                    pass

                cf.disconnect(robot)
                robot = None
                continue

            ip_in = input(f"로봇 IP 입력 (엔터=기본 {robot_ip}) > ").strip()
            if ip_in:
                robot_ip = ip_in

            try:
                robot = cf.connect(robot_ip)

                # ✅ 연결되면 비전 스트리밍 ON
                try:
                    mb.start_stream()
                except Exception as e:
                    print(f"[WARN] Vision stream start failed: {e}")

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
            # ✅ 스트리밍 유지한 채 "지금부터 10프레임 평균" 요청
            if not mb.is_running():
                print("[WARN] 비전 스트리밍이 꺼져있습니다. (0번 연결하면 자동 ON)\n")
                continue

            print("\n[ACTION] Vision measure request (avg 10 frames)...")
            try:
                res = mb.measure_avg(n=10)  # dict or None
            except Exception as e:
                print(f"[ERROR] Vision measure 예외: {e}\n")
                res = None

            if res is None:
                print("[FAIL] 측정 실패(None) (timeout or no valid samples)\n")
                continue

            last_measure = res
            print("[OK] 측정 결과 ✅ (last_measure 저장됨)")
            _print_measure(res)
            print()

        elif cmd == "3":
            if last_tcp_pose is None or last_measure is None:
                print("\n[WARN] 1,2번 먼저 실행해야 합니다. (last_tcp_pose / last_measure 필요)\n")
                continue

            try:
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
                    last_target_pose = tp.build_target_pose(last_tcp_pose, last_measure)

                    print("\n[OK] target_pose 생성/저장 ✅ (orientation 유지)")
                    print("current_pose :", tp.fmt_pose6(last_tcp_pose))
                    print("target_pose  :", tp.fmt_pose6(last_target_pose))

            except Exception as e:
                print(f"\n[ERROR] target_pose 생성 실패: {e}\n")
                continue

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

        elif cmd == "11":
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

        elif cmd == "12":
            # ✅ 자동 반복 루프 (Home -> Measure -> Calc -> IK Check -> Pick -> ★Home★ -> Place)
            if robot is None:
                print("[WARN] 로봇 연결 필요\n")
                continue
            if initial_joint6 is None:
                print("[WARN] 1번을 눌러 초기 위치(Home)를 먼저 저장하세요!\n")
                continue
            if not mb.is_running():
                print("[WARN] 카메라가 꺼져있습니다. 0번을 누르세요.\n")
                continue

            raw_n = input("반복할 횟수를 입력하세요 (예: 4) > ").strip()
            try:
                loop_cnt = int(raw_n)
            except:
                print("[ERR] 숫자가 아닙니다.")
                continue

            print(f"\n🚀 자동 작업 시작: {loop_cnt}회 반복 예정")
            
            for i in range(loop_cnt):
                print(f"\n========================================")
                print(f" [Cycle {i+1} / {loop_cnt}] 작업 시작")
                print(f"========================================")

                # 1. 초기 위치 복귀
                print(f" -> [Step 1] 초기 위치 이동...")
                out_home = rh.cmd7_return_to_initial(robot, initial_joint6, reconnect)
                if not out_home["ok"]:
                    print(f"❌ [Step 1] 실패: {out_home['msg']}")
                    break
                time.sleep(1.0) 
                
                # 2. 현재 로봇 포즈 읽기
                (e1, cur_p), (e2, cur_j) = rs.read_pose_joint(robot, reconnect)
                if e1!=0 or e2!=0: 
                    print("❌ 포즈 읽기 실패")
                    break

                # 3. 비전 측정
                print(f" -> [Step 2] 비전 측정 중...")
                res = mb.measure_avg(n=10)
                if res is None:
                    print("❌ [Step 2] 박스를 찾지 못함 -> 루프 중단")
                    break
                _print_measure(res)

                # 4. Target 생성
                print(f" -> [Step 3] 목표 좌표 계산...")
                try:
                    if hasattr(tp, "build_target_pose_with_debug"):
                        out_tp = tp.build_target_pose_with_debug(cur_p, res)
                        target = out_tp["target_pose6"]
                        # [수정] 문제의 함수 호출 제거하고 단순 출력으로 대체
                        print(f"    (Preview) Target: {tp.fmt_pose6(target)}")
                    else:
                        target = tp.build_target_pose(cur_p, res)
                except Exception as e:
                    print(f"❌ [Step 3] 계산 에러: {e}")
                    break

                # 5. IK 점검 및 보정
                print(f" -> [Step 4] IK 점검 및 보정...")
                ik_out = ika.cmd4_check_and_adjust_target_only(
                    robot=robot,
                    reconnect=reconnect,
                    cur_pose6=cur_p,
                    cur_joint6=cur_j,
                    target_pose6=target
                )
                
                if not ik_out.get("ok", False):
                    print(f"❌ [Step 4] IK 실패: {ik_out.get('msg')}")
                    break
                
                target = ik_out["target"]
                if ik_out.get("adjusted", False):
                    print(f"    (IK 보정됨) {rs.fmt_pose6(target)}")
                else:
                    print("    (IK 정상)")

                # 6. Smooth Auto (Pick)
                print(f" -> [Step 5] Pick (Smooth Auto)...")
                out_pick = sa.cmd9_smooth_auto(
                    robot=robot,
                    reconnect=reconnect,
                    last_target_pose6=target,
                    last_measure=res,
                    state=state,
                    tool=0, user=0,
                    auto_grip_close=True
                )
                if not out_pick["ok"]:
                    print(f"❌ [Step 5] Pick 실패: {out_pick['msg']}")
                    break
                
                # ========================================================
                # ✅ [추가됨] Pick 후 안전하게 Home 복귀 (Step 5.5)
                # ========================================================
                print(f" -> [Step 5.5] 안전을 위해 Home으로 복귀...")
                out_safe_home = rh.cmd7_return_to_initial(robot, initial_joint6, reconnect)
                if not out_safe_home["ok"]:
                    print(f"❌ [Step 5.5] Home 복귀 실패: {out_safe_home['msg']}")
                    break
                time.sleep(0.5)
                # ========================================================

                # 7. Stack Cycle (Place)
                print(f" -> [Step 6] Place (Stacking)...")
                out_place = sc.cmd11_stack_cycle(
                    robot=robot,
                    reconnect=reconnect,
                    state=state,
                    home_joint6=initial_joint6,
                    tool=0, user=0
                )
                if not out_place["ok"]:
                    print(f"❌ [Step 6] Place 실패: {out_place['msg']}")
                    break
                
                print(f"✅ Cycle {i+1} 완료! (현재 적재 수: {state['stack_counter']})")
                time.sleep(0.5)

            print("\n🎉 모든 반복 작업 종료.\n")

        elif cmd == "q":
            print("[EXIT] 종료합니다.")
            break

        else:
            print("[WARN] 잘못된 입력입니다.")

    # cleanup
    try:
        mb.stop_stream()
    except Exception:
        pass

    if robot is not None:
        cf.disconnect(robot)

if __name__ == "__main__":
    main()