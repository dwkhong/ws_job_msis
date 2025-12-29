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
    if not isinstance(j, (list, tuple)):
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


# -------------------------
# ✅ (1번) 현재 상태 읽기
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
