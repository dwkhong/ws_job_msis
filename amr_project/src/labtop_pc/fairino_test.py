import sys
import time

# ✅ Robot.cp39-win_amd64.pyd 가 있는 폴더를 Python 경로에 추가
sys.path.insert(
    0,
    r"C:\Users\rhdeh\ws_job_msis\amr_project\driver\fairino-python-sdk-main\windows\fairino\build\lib.win-amd64-cpython-39"
)
# ✅ 여기서는 "fairino 패키지"가 아니라 "Robot 모듈(.pyd)"을 직접 import 해야 함
import Robot

# Establish a connection with the robot controller and return a robot object if the connection is successful
robot = Robot.RPC("192.168.58.3")

# --- 테스트: SDK 버전 출력(가능한 경우) ---
if hasattr(robot, "GetSDKVersion"):
    print("SDK Version:", robot.GetSDKVersion())
elif hasattr(Robot, "GetSDKVersion"):
    print("SDK Version:", Robot.GetSDKVersion())
else:
    print("[WARN] GetSDKVersion not found. Available candidates:",
          [m for m in dir(robot) if "Version" in m or "SDK" in m][:50])

time.sleep(1)
