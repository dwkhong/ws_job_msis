import sys
import time

# (너 코드에 있던 import 유지)
from box_test_3 import main as measure_main  # 지금은 사용 안 해도 OK

# ✅ Robot.cp39-win_amd64.pyd 폴더를 Python 경로에 추가
sys.path.insert(
    0,
    r"C:\Users\rhdeh\ws_job_msis\amr_project\driver\fairino-python-sdk-main\windows\fairino\build\lib.win-amd64-cpython-39"
)
import Robot

ROBOT_IP = "192.168.58.3"

# -------------------------
# Config
# -------------------------
TOOL = 0
USER = 0
VEL_JOG = 10.0      # MoveJ 속도(%) - 안전하게 낮게
BLENDT = -1.0       # -1 blocking


# -------------------------
# Global cache (저장용)
# -------------------------
LAST_JOINT_DEG = None
LAST_TCP_POSE  = None


def connect_robot(ip: str):
    """로봇 연결"""
    robot = Robot.RPC(ip)
    return robot


def safe_close(robot):
    try:
        robot.CloseRPC()
    except Exception:
        pass


def try_get_tcp_pose(robot):
    """
    SDK 버전에 따라 TCP pose 함수명이 다를 수 있어서,
    존재하는 함수가 있으면 시도해보고 없으면 None 반환.
    """
    # 후보 함수명들 (SDK에 따라 다름)
    candidates = [
        ("GetActualTCPPose", (0,)),          # (flag) 형태일 가능성
        ("GetActualToolFlangePose", (0,)),
        ("GetActualPose", (0,)),
        ("GetActualCartPos", (0,)),
        ("GetActualTCPPos", (0,)),
    ]

    for fn, args in candidates:
        if hasattr(robot, fn):
            try:
                ret, pose = getattr(robot, fn)(*args)
                if ret == 0:
                    return pose
            except TypeError:
                # 인자 형태가 다를 수 있음 → 인자 없이도 한 번 시도
                try:
                    ret, pose = getattr(robot, fn)()
                    if ret == 0:
                        return pose
                except Exception:
                    pass
            except Exception:
                pass

    return None


def get_current_state(robot):
    """
    현재 joint(deg) / tcp_pose 읽어서 리턴
    """
    global LAST_JOINT_DEG, LAST_TCP_POSE

    flag = 0
    ret_j, joints = robot.GetActualJointPosDegree(flag)
    if ret_j != 0:
        raise RuntimeError(f"GetActualJointPosDegree errcode={ret_j}")

    tcp_pose = try_get_tcp_pose(robot)

    LAST_JOINT_DEG = joints
    LAST_TCP_POSE = tcp_pose
    return joints, tcp_pose


def print_state(joints, tcp_pose):
    j_str = ", ".join([f"{v:.3f}" for v in joints[:6]])
    print("\n================ CURRENT STATE ================")
    print(f"JOINT(deg) : [{j_str}]")

    if tcp_pose is None:
        print("TCP pose   : (SDK에서 pose 읽는 함수 못 찾음 / 지원 안 될 수 있음)")
    else:
        # pose는 보통 [x,y,z,rx,ry,rz] mm/deg 형태
        try:
            p_str = ", ".join([f"{v:.3f}" for v in tcp_pose[:6]])
            print(f"TCP pose   : [{p_str}]")
        except Exception:
            print(f"TCP pose   : {tcp_pose}")
    print("================================================\n")


def move_j6_relative(robot, delta_deg: float):
    """
    현재 joint 읽고, j6만 delta_deg 만큼 변경해서 MoveJ 수행
    """
    flag = 0
    ret_j, j_now = robot.GetActualJointPosDegree(flag)
    if ret_j != 0:
        print(f"GetActualJointPosDegree errcode={ret_j}")
        return ret_j

    j_tgt = list(j_now)
    j_tgt[5] = float(j_tgt[5]) + float(delta_deg)   # ✅ J6만 변경

    print(f"[MOVEJ] J6: {j_now[5]:.3f} -> {j_tgt[5]:.3f} (delta {delta_deg:+.3f} deg)")

    err = robot.MoveJ(
        joint_pos=j_tgt,
        tool=TOOL,
        user=USER,
        vel=VEL_JOG,
        blendT=BLENDT
    )
    print(f"MoveJ errcode: {err}")

    return err


def prompt_menu():
    print("=======================================")
    print("무슨 기능을 할까요?")
    print("  1 : 현재 joint(deg) + (가능하면) tcp_pose 가져오기/저장")
    print("  2 : J6 상대 이동 (예: +30 / -30 입력하면 J6만 그만큼 이동)")
    print("  q : 종료")
    print("=======================================")
    return input("입력 (1/2/q) > ").strip()


def prompt_delta():
    """
    +30 / -30 / 15 같은 입력을 float로 파싱
    """
    s = input("J6 변화량 입력 (deg, 예: +30 / -30) > ").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        print("숫자로 입력해줘. 예: +30 또는 -30")
        return None


def main():
    robot = None
    try:
        robot = connect_robot(ROBOT_IP)
        print(f"[OK] Connected to robot: {ROBOT_IP}")

        while True:
            sel = prompt_menu()

            if sel.lower() == "q":
                break

            elif sel == "1":
                try:
                    joints, tcp_pose = get_current_state(robot)
                    print_state(joints, tcp_pose)
                except Exception as e:
                    print(f"[ERR] state read failed: {e}")

            elif sel == "2":
                delta = prompt_delta()
                if delta is None:
                    continue
                move_j6_relative(robot, delta)

            else:
                print("지원하지 않는 메뉴야. (1/2/q) 중에서 선택해줘.")

    finally:
        if robot is not None:
            safe_close(robot)
        print("[DONE] Closed connection.")


if __name__ == "__main__":
    main()
