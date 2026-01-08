# main.py
from app_config import ROBOT_IP_DEFAULT

from robot import connect_fairino as cf
from robot import robot_state as rs
from robot import target_pose as tp
from robot import ik_check as ik
from robot import smooth_auto as sa
from robot import return_home as rh
from robot import gripper_control as gc
from robot import j4_rotate as j4r
from robot import j6_rotate as j6r
from robot import auto_pick_place as ap

from vision import measure_box_2 as mb


def print_menu():
    print("\n===== Main Menu =====")
    print("0. Robot Connect/Disconnect")
    print("1. Vision On/Off")
    print("2. Read Robot Pose/Joint (cache)")
    print("3. Vision Measure Avg (cache)")
    print("4. Build Target Pose (use last 2+3)")
    print("5. IK Check (use last 4 + last joint)")
    print("6. Smooth move 1-step ")
    print("7. Smooth move 2-step ")
    print("8. Return Home + Gripper Open")
    print("9. Gripper Control Menu")
    print("10. J4 Rotate (input delta deg)")
    print("11. J6 Rotate (input delta deg)")
    print("12. Auto Pick&Place (N cycles)")
    print("q. Quit")


def _reconnect():
    cf.connect(ROBOT_IP_DEFAULT)


def main():
    while True:
        print_menu()
        cmd = input("입력 (0/1/2/3/4/5/6/7/8/9/10/11/q) > ").strip().lower()

        if cmd == "q":
            break

        if cmd == "0":
            cf.toggle(ROBOT_IP_DEFAULT)

        elif cmd == "1":
            mb.toggle_stream()

        elif cmd == "2":
            rs.cmd_read_pose_joint(cf.get_robot(), reconnect=_reconnect)

        elif cmd == "3":
            mb.cmd_measure_avg()

        elif cmd == "4":
            tp.cmd_build_target_from_last(cf.get_robot(), reconnect=_reconnect, use_last_pose=True)

        elif cmd == "5":
            ik.cmd_check_target_from_last(cf.get_robot(), reconnect=_reconnect)

        elif cmd == "6":
            sa.cmd6(cf.get_robot(), reconnect=_reconnect)

        elif cmd == "7":
            sa.cmd7(cf.get_robot(), reconnect=_reconnect)

        elif cmd == "8":
            out = rh.cmd8(cf.get_robot(), reconnect=_reconnect)
            if not out.get("ok", False):
                print("[8] FAIL:", out.get("msg", ""))

        elif cmd == "9":
            gc.cmd9(cf.get_robot(), reconnect=_reconnect)

        elif cmd == "10":
            j4r.cmd10(cf.get_robot(), reconnect=_reconnect)

        elif cmd == "11":
            j6r.cmd11(cf.get_robot(), reconnect=_reconnect)
        
        elif cmd == "12":
            ap.cmd12(cf.get_robot(), reconnect=_reconnect)

        else:
            pass

if __name__ == "__main__":
    main()


