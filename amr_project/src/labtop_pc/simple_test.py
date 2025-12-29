# j4_test_menu.py
# ============================================================
# ✅ 테스트용 단일 파일 (j6_rotate.py 스타일)
# 0 : 로봇 연결/해제 (토글)
# 1 : 현재 tcp_pose + joint(deg) 읽기
# 2 : J4만 delta(deg) 만큼 MoveJ
#
# - GetActualJointPosDegree(flag=1) 사용
# - MoveJ는 "키워드 인자"로 호출 (너 j6_rotate에서 되는 방식 그대로)
# ============================================================

import sys
import time
from typing import Any, List, Optional, Tuple

# -------------------------
# 사용자 환경
# -------------------------
FAIRINO_PYD_DIR = r"C:\Users\rhdeh\ws_job_msis\amr_project\driver\fairino-python-sdk-main\windows\fairino\build\lib.win-amd64-cpython-310"
ROBOT_IP = "192.168.0.15"

TOOL_ID = 0
USER_ID = 0

# MoveJ 속도(0~100 퍼센트)
MOVEJ_VEL = 20.0
MOVEJ_BLENDT = -1.0

# RPC retry
RPC_RETRY = 1
RPC_RETRY_SLEEP_SEC = 0.25

# 안전 클램프
J4_MAX_STEP_DEG = 45.0  # 한 번에 최대 ±45도
J4_INPUT_LIMIT_DEG = 60.0  # 입력 제한


# -------------------------
# Utils
# -------------------------
def fmt_pose6(p: List[float]) -> str:
    x, y, z, rx, ry, rz = p[:6]
    return f"[x,y,z,rx,ry,rz]=[{x:.3f}, {y:.3f}, {z:.3f}, {rx:.3f}, {ry:.3f}, {rz:.3f}]"


def fmt_joint(j):
    if not isinstance(j, (list, tuple)):
        return str(j)
    return "[" + ", ".join(f"{float(v):.3f}" for v in j[:6]) + "]"


def ensure_joint6(j):
    if not isinstance(j, (list, tuple)) or len(j) < 6:
        raise ValueError(
            f"joint must be list/tuple len>=6, got {type(j)} "
            f"len={len(j) if hasattr(j, '__len__') else 'N/A'}"
        )
    return [float(x) for x in j[:6]]


def clamp(v, lo, hi):
    v = float(v)
    return max(float(lo), min(float(hi), v))


def safe_call(fn, *args, retry=1, sleep_sec=0.25, reconnect_cb=None, **kwargs):
    last_e = None
    for k in range(retry + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_e = e
            msg = str(e).lower()
            if ("timed out" in msg) or ("timeout" in msg) or ("实时数据失败" in str(e)):
                print(f"[WARN] RPC timeout-like error: {e}")
                if reconnect_cb is not None:
                    try:
                        reconnect_cb()
                    except Exception as e2:
                        print(f"[WARN] reconnect failed: {e2}")
            if k < retry:
                time.sleep(sleep_sec)
                continue
            raise last_e


def _unwrap_pose_or_joint(res: Any) -> Optional[List[float]]:
    # (err, data)
    if isinstance(res, (tuple, list)) and len(res) >= 2 and isinstance(res[1], (tuple, list)):
        data = res[1]
        if len(data) >= 6:
            return [float(v) for v in list(data)[:6]]
        return None
    # data only
    if isinstance(res, (tuple, list)) and len(res) >= 6:
        return [float(v) for v in list(res)[:6]]
    return None


def connect_robot(ip: str):
    if FAIRINO_PYD_DIR and FAIRINO_PYD_DIR not in sys.path:
        sys.path.insert(0, FAIRINO_PYD_DIR)

    # 1) fairino 패키지 형태
    try:
        from fairino import Robot as FairinoRobot  # type: ignore
        return FairinoRobot.RPC(ip)
    except Exception:
        pass

    # 2) Robot.pyd 직접 import 형태
    try:
        import Robot  # type: ignore
        return Robot.RPC(ip)
    except Exception as e:
        raise RuntimeError("Robot SDK import 실패. FAIRINO_PYD_DIR / Python 버전 / pyd 경로 확인") from e


def close_robot(robot) -> None:
    try:
        if robot is None:
            return
        if hasattr(robot, "CloseRPC"):
            robot.CloseRPC()
    except Exception:
        pass


def get_tcp_pose(robot) -> List[float]:
    for name in ["GetActualTCPPose", "GetActualToolFlangePose", "GetActualTCP"]:
        if hasattr(robot, name):
            res = safe_call(getattr(robot, name), retry=RPC_RETRY, sleep_sec=RPC_RETRY_SLEEP_SEC)
            pose = _unwrap_pose_or_joint(res)
            if pose is not None:
                return pose
    raise RuntimeError("tcp_pose 읽기 함수가 없습니다. (GetActualTCPPose 등 확인)")


def get_joints_deg(robot) -> List[float]:
    # ✅ 너가 쓰던 형태: GetActualJointPosDegree(flag=1)
    if hasattr(robot, "GetActualJointPosDegree"):
        err, data = safe_call(
            robot.GetActualJointPosDegree,
            flag=1,
            retry=RPC_RETRY,
            sleep_sec=RPC_RETRY_SLEEP_SEC,
        )
        if int(err) != 0:
            raise RuntimeError(f"GetActualJointPosDegree err={err}")
        return ensure_joint6(data)

    # fallback
    for name in ["GetActualJointPos"]:
        if hasattr(robot, name):
            res = safe_call(getattr(robot, name), retry=RPC_RETRY, sleep_sec=RPC_RETRY_SLEEP_SEC)
            j = _unwrap_pose_or_joint(res)
            if j is not None:
                return j
    raise RuntimeError("joint 읽기 함수가 없습니다. (GetActualJointPosDegree 등 확인)")


# -------------------------
# J4 MoveJ (j6_rotate 스타일)
# -------------------------
def movej_j4_delta(robot, delta_deg: float, tool=0, user=0, vel=None, blendT=None):
    if vel is None:
        vel = MOVEJ_VEL
    if blendT is None:
        blendT = MOVEJ_BLENDT

    # 안전 클램프
    delta_deg = clamp(delta_deg, -J4_MAX_STEP_DEG, +J4_MAX_STEP_DEG)

    j_now = get_joints_deg(robot)
    j_tgt = list(j_now)
    j_tgt[5] = float(j_tgt[5]) + float(delta_deg)  # ✅ J4

    print(f"[MOVEJ-J4] cur  joint: {fmt_joint(j_now)}")
    print(f"[MOVEJ-J4] tgt  joint: {fmt_joint(j_tgt)}")
    print(f"[MOVEJ-J4] J4: {j_now[3]:.3f} -> {j_tgt[3]:.3f} (delta {delta_deg:+.3f} deg)")

    # ✅ 너 환경에서 잘 되는 키워드 호출 패턴 유지
    ret = safe_call(
        robot.MoveJ,
        joint_pos=j_tgt,
        tool=int(tool),
        user=int(user),
        vel=float(vel),
        blendT=float(blendT),
        retry=RPC_RETRY,
        sleep_sec=RPC_RETRY_SLEEP_SEC,
    )
    # ret가 tuple일 수도 있고 int일 수도 있음
    if isinstance(ret, (tuple, list)) and len(ret) >= 1:
        errcode = int(ret[0])
    else:
        errcode = int(ret)

    print(f"[RET] MoveJ(J4) errcode: {errcode}")
    return errcode


def print_menu(connected: bool):
    print("\n=======================================")
    print("J4 MoveJ 테스트 (단일 파일)")
    print(f"  연결 상태: {'CONNECTED' if connected else 'DISCONNECTED'}")
    print("  0 : 로봇 연결/해제 (토글)")
    print("  1 : 현재 tcp_pose + joint(deg) 읽기")
    print("  2 : J4 MoveJ(delta deg)")
    print("  q : 종료")
    print("=======================================")


def main():
    robot = None

    while True:
        print_menu(robot is not None)
        sel = input("선택: ").strip().lower()

        if sel in ("q", "quit", "exit"):
            break

        if sel == "0":
            if robot is None:
                try:
                    robot = connect_robot(ROBOT_IP)
                    print(f"[OK] 연결 성공: {ROBOT_IP}")
                except Exception as e:
                    robot = None
                    print(f"[FAIL] 연결 실패: {e}")
            else:
                close_robot(robot)
                robot = None
                print("[OK] 연결 해제")
            continue

        if robot is None:
            print("[WARN] 먼저 0번으로 연결하세요.")
            continue

        if sel == "1":
            try:
                p = get_tcp_pose(robot)
                j = get_joints_deg(robot)
                print(f"[TCP_POSE] {fmt_pose6(p)}")
                print(f"[JOINT]    {fmt_joint(j)}")
            except Exception as e:
                print(f"[FAIL] 읽기 실패: {e}")
            continue

        if sel == "2":
            s = input(f"J4 delta 입력 (deg, 입력 제한 ±{J4_INPUT_LIMIT_DEG}): ").strip()
            try:
                d = float(s)
            except Exception:
                print("[WARN] 숫자를 입력하세요.")
                continue

            d = clamp(d, -J4_INPUT_LIMIT_DEG, +J4_INPUT_LIMIT_DEG)

            try:
                err = movej_j4_delta(robot, d, tool=TOOL_ID, user=USER_ID)
                if err == 0:
                    print("[OK] J4 MoveJ 완료")
                    # 이동 후 상태 출력
                    p2 = get_tcp_pose(robot)
                    j2 = get_joints_deg(robot)
                    print(f"[AFTER_POSE]  {fmt_pose6(p2)}")
                    print(f"[AFTER_JOINT] {fmt_joint(j2)}")
                else:
                    print(f"[FAIL] MoveJ errcode={err}")
            except Exception as e:
                print(f"[FAIL] MoveJ 실패: {e}")
            continue

        print("[WARN] 0/1/2/q 중에서 선택하세요.")

    close_robot(robot)
    print("bye")


if __name__ == "__main__":
    main()




