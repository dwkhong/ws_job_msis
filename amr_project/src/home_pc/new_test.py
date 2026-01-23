#!/usr/bin/env python3
from config.app_config import FAIRINO_PYD_PATH, ROBOT_IP_DEFAULT
from robot.robot_connector import RobotConnector

HOME_CART = [329.766, -101.959, 802.910, -179.919, 0.280, 179.344]
T1_CART   = [84.135, -445.200, 714.081, 179.285, 0.288, 87.147]
T2_CART   = [-131.148, -507.733, 659.857, 179.360, -3.263, 89.044]

def connect_robot(ip: str = ROBOT_IP_DEFAULT):
    conn = RobotConnector(ip=ip, sdk_path=FAIRINO_PYD_PATH)
    if not conn.connect():
        raise RuntimeError("Robot connect failed")
    return conn.get_robot(), conn

def main():
    robot, conn = connect_robot()
    try:
        ret = robot.MoveCart(desc_pos=T2_CART, tool=0, user=0, vel=80, blendT=100)
        print(f"MoveCart ret={ret}")
    finally:
        conn.disconnect()

if __name__ == "__main__":
    main()
