# robot/robot_state.py
import time
from .robot_config import RPC_RETRY, RPC_RETRY_SLEEP_SEC

# -------------------------
# Utils (출력/검증)
# -------------------------
def fmt_pose6(pose):
    if not isinstance(pose, (list, tuple)) or len(pose) < 6:
        return str(pose)
    x, y, z, rx, ry, rz = pose[:6]
    return f"[x,y,z,rx,ry,rz]=[{x:.3f}, {y:.3f}, {z:.3f}, {rx:.3f}, {ry:.3f}, {rz:.3f}]"


def fmt_joint(j):
    if not isinstance(j, (list, tuple)) or len(j) < 6:
        return str(j)
    return "[" + ", ".join(f"{float(v):.3f}" for v in j[:6]) + "]"


def ensure_pose6(p):
    if not isinstance(p, (list, tuple)) or len(p) < 6:
        raise ValueError(f"pose invalid: {p}")
    return [float(x) for x in p[:6]]


def ensure_joint6(j):
    if not isinstance(j, (list, tuple)) or len(j) < 6:
        raise ValueError(f"joint invalid: {j}")
    return [float(x) for x in j[:6]]


# -------------------------
# Safe call (timeout 대응)
# -------------------------
def safe_call(fn, *args, retry=RPC_RETRY, sleep_sec=RPC_RETRY_SLEEP_SEC, reconnect_cb=None, **kwargs):
    last_e = None

    for k in range(retry + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_e = e
            msg = str(e).lower()

            timeout_like = ("timed out" in msg) or ("timeout" in msg) or ("实时数据失败" in str(e))

            # retry가 남아있으면: 잠깐 쉬고 재시도
            if k < retry:
                if timeout_like:
                    print(f"[WARN] RPC timeout-like error (retry {k+1}/{retry}): {e}")
                time.sleep(sleep_sec)
                continue

            # 마지막 시도까지 실패
            if timeout_like:
                print(f"[WARN] RPC timeout-like error (final): {e}")
                if reconnect_cb is not None:
                    try:
                        reconnect_cb()
                    except Exception as e2:
                        print(f"[WARN] reconnect failed: {e2}")

            raise last_e


# -------------------------
# ✅ 현재 상태 읽기 (pose + joint)
# -------------------------
def read_pose_joint(robot, reconnect=None):
    """
    return:
      (err_p, pose6 or None), (err_j, joint6 or None)
    """
    err_p, pose = safe_call(robot.GetActualTCPPose, flag=1, reconnect_cb=reconnect)
    err_j, joint = safe_call(robot.GetActualJointPosDegree, flag=1, reconnect_cb=reconnect)

    if err_p != 0 or err_j != 0:
        return (err_p, None), (err_j, None)

    return (0, ensure_pose6(pose)), (0, ensure_joint6(joint))


def read_joint6(robot, reconnect=None):
    """
    pose 없이 joint만 읽기 (조용히 초기값 저장할 때 사용)
    return: (err, joint6 or None)
    """
    err_j, joint = safe_call(robot.GetActualJointPosDegree, flag=1, reconnect_cb=reconnect)
    if int(err_j) != 0 or joint is None:
        return int(err_j), None
    return 0, ensure_joint6(joint)


# -------------------------
# ✅ main에서 print 제거용 래퍼 (2번)
# -------------------------
_LAST_POSE = None
_LAST_JOINT = None

# ✅ "초기(Home)" joint 캐시 (1회만 저장)
_INITIAL_JOINT = None


def set_initial_joint6(joint6, force: bool = False):
    global _INITIAL_JOINT
    if joint6 is None:
        return
    if force or (_INITIAL_JOINT is None):
        _INITIAL_JOINT = ensure_joint6(joint6)


def get_initial_joint6():
    return _INITIAL_JOINT


def try_capture_initial_joint(robot, reconnect=None, force: bool = False, verbose: bool = True):
    """
    ✅ 초기 joint가 없으면(또는 force=True) 현재 joint를 읽어서 초기값으로 저장
    - pose 출력 없이 joint만 읽음
    """
    global _INITIAL_JOINT
    if (not force) and (_INITIAL_JOINT is not None):
        return _INITIAL_JOINT

    if robot is None:
        return None

    err, j = read_joint6(robot, reconnect=reconnect)
    if err == 0 and j is not None:
        set_initial_joint6(j, force=force)
        if verbose:
            print("[INIT] initial_joint6 saved:", fmt_joint(_INITIAL_JOINT))
        return _INITIAL_JOINT

    if verbose:
        print(f"[INIT] initial_joint6 capture failed err={err}")
    return None


def cmd_read_pose_joint(robot, reconnect=None):
    global _LAST_POSE, _LAST_JOINT

    if robot is None:
        print("[WARN] 로봇이 연결되어 있지 않습니다. (0번으로 연결)")
        return None

    (err_p, pose), (err_j, joint) = read_pose_joint(robot, reconnect=reconnect)

    if err_p == 0 and pose is not None:
        _LAST_POSE = pose
        print("tcp_pose :", fmt_pose6(pose))
    else:
        print(f"[ERR] GetActualTCPPose failed: err={err_p}")

    if err_j == 0 and joint is not None:
        _LAST_JOINT = joint

        # ✅ 초기값이 비어있으면 "한 번만" 저장
        set_initial_joint6(joint, force=False)

        print("joint6   :", fmt_joint(joint))
    else:
        print(f"[ERR] GetActualJointPosDegree failed: err={err_j}")

    return {
        "err_pose": err_p,
        "pose": pose,
        "err_joint": err_j,
        "joint": joint,
    }


def get_last_pose_joint():
    """(pose6, joint6) 캐시 반환"""
    return _LAST_POSE, _LAST_JOINT


def get_last_joint6():
    """joint6 캐시만 반환"""
    return _LAST_JOINT

