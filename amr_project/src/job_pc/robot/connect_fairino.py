# connect_fairino.py
import sys
from app_config import FAIRINO_PYD_PATH

# ============================================================
# ✅ Fairino Robot .pyd 경로 주입 (공용 config)
# ============================================================
if FAIRINO_PYD_PATH not in sys.path:
    sys.path.insert(0, FAIRINO_PYD_PATH)

import Robot  # Robot.cp39-win_amd64.pyd


def connect(ip: str):
    print(f"[ROBOT] Connecting... ip={ip}")
    rb = Robot.RPC(ip)
    print("[ROBOT] Connected ✅")
    return rb


def disconnect(robot):
    if robot is None:
        return
    print("[ROBOT] Closing...")
    try:
        robot.CloseRPC()
    except Exception:
        pass
    print("[ROBOT] Closed ✅")
