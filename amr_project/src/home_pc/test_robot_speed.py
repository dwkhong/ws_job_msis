#!/usr/bin/env python3
import time
from config.app_config import FAIRINO_PYD_PATH, ROBOT_IP_DEFAULT
from robot.robot_connector import RobotConnector

# ==========================
# 3 포인트 (네가 준 값)
# ==========================
# MoveCart: [x, y, z, rx, ry, rz]
HOME_CART = [329.766, -101.959, 802.910, -179.919, 0.280, 179.344]
T1_CART   = [84.135, -445.200, 714.081, 179.285, 0.288, 87.147]
T2_CART   = [-131.148, -507.733, 659.857, 179.360, -3.263, 89.044]

# MoveJ: [j1, j2, j3, j4, j5, j6]
HOME_JOINT = [0.033, -90.450, -34.076, -145.194, 89.915, 90.689]
T1_JOINT   = [-66.404, -93.019, -50.178, -126.227, 90.512, 116.450]
T2_JOINT   = [-93.365, -100.174, -50.207, -122.906, 90.502, 87.595]

# ==========================
# 속도(미리 고정)
# ==========================
VEL_CART  = 100.0
VEL_JOINT = 100.0

# ==========================
# 고정 파라미터
# ==========================
TOOL = 0
USER = 0
BLENDT = 100.0

# non-blocking이라 너무 빨리 쌓이지 않게 텀
SLEEP_BETWEEN = 0.2


# ============================================================
# 연결 파트 "분리"
# ============================================================
def connect_robot(ip: str = ROBOT_IP_DEFAULT):
    conn = RobotConnector(ip=ip, sdk_path=FAIRINO_PYD_PATH)
    if not conn.connect():
        raise RuntimeError("Robot connect failed")
    return conn.get_robot(), conn


def disconnect_robot(conn: RobotConnector):
    try:
        conn.disconnect()
    except Exception:
        pass


# ============================================================
# 속도(이동) 테스트
# ============================================================
def run_movecart(robot):
    path = [("HOME", HOME_CART), ("T1", T1_CART), ("T2", T2_CART), ("HOME", HOME_CART)]
    for name, pose in path:
        print(f"[MoveCart] -> {name} vel={VEL_CART}")
        ret = robot.MoveCart(desc_pos=pose, tool=TOOL, user=USER, vel=VEL_CART, blendT=BLENDT)
        print(f"  ret={ret}")
        time.sleep(SLEEP_BETWEEN)


def run_movej(robot):
    path = [("HOME", HOME_JOINT), ("T1", T1_JOINT), ("T2", T2_JOINT), ("HOME", HOME_JOINT)]
    for name, joint in path:
        print(f"[MoveJ] -> {name} vel={VEL_JOINT}")
        ret = robot.MoveJ(joint_pos=joint, tool=TOOL, user=USER, vel=VEL_JOINT, blendT=BLENDT)
        print(f"  ret={ret}")
        time.sleep(SLEEP_BETWEEN)


def main():
    robot, conn = connect_robot()
    try:
        sel = input("Select: 1=MoveCart, 2=MoveJ, 0=Both : ").strip()
        if sel == "1":
            run_movecart(robot)
        elif sel == "2":
            run_movej(robot)
        else:
            run_movecart(robot)
            run_movej(robot)
    finally:
        disconnect_robot(conn)


if __name__ == "__main__":
    main()
