# robot/connect_fairino.py
import sys
from app_config import FAIRINO_PYD_PATH

if FAIRINO_PYD_PATH not in sys.path:
    sys.path.insert(0, FAIRINO_PYD_PATH)

import Robot  # noqa

_robot = None
_robot_ip = None

def is_connected() -> bool:
    return _robot is not None

def get_robot():
    return _robot

def connect(ip: str):
    global _robot, _robot_ip

    if _robot is not None:
        print(f"[ROBOT] Already connected ✅ ip={_robot_ip}")
        return _robot

    print(f"[ROBOT] Connecting... ip={ip}")
    _robot = Robot.RPC(ip)
    _robot_ip = ip
    print("[ROBOT] Connected ✅")
    return _robot

def disconnect():
    global _robot, _robot_ip

    if _robot is None:
        print("[ROBOT] Not connected.")
        return

    print("[ROBOT] Closing...")
    try:
        _robot.CloseRPC()
    except Exception:
        pass

    _robot = None
    _robot_ip = None
    print("[ROBOT] Closed ✅")


def toggle(ip: str):
    if is_connected():
        disconnect()
        return None
    return connect(ip)


