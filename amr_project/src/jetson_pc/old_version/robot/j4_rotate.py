# robot/j4_rotate.py
import time

# ✅ robot_config를 단일 소스로 사용
try:
    from . import robot_config as rc
except Exception:
    import robot_config as rc


# -------------------------
# Utils (module local)
# -------------------------
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


# -------------------------
# Core (J4)
# -------------------------
def rotate_j4_by_delta(
    robot,
    delta_deg: float,
    tool=0,
    user=0,
    vel: float | None = None,
    blendT: float | None = None,
    reconnect=None,
    max_step_deg: float | None = None,
):
    """
    현재 조인트를 읽어서 J4만 delta_deg만큼 더한 뒤 MoveJ 수행
    반환: errcode(int)

    - 기본 vel/blendT는 rc에서 읽음(없으면 기본값 사용)
      * vel    = rc.MOVEJ_VEL_J4 (없으면 60.0)
      * blendT = rc.MOVEJ_BLENDT_J4 (없으면 -1.0)
    - (선택) max_step_deg로 delta clamp 가능
      * max_step_deg = rc.J4_MAX_STEP_DEG (없으면 None=클램프 안함)
    """
    if vel is None:
        vel = float(getattr(rc, "MOVEJ_VEL_J4", 60.0))
    if blendT is None:
        blendT = float(getattr(rc, "MOVEJ_BLENDT_J4", -1.0))

    if max_step_deg is None:
        max_step_deg = getattr(rc, "J4_MAX_STEP_DEG", None)
    if max_step_deg is not None:
        delta_deg = clamp(delta_deg, -float(max_step_deg), +float(max_step_deg))

    err_j, cur_joint = safe_call(robot.GetActualJointPosDegree, flag=1, retry=1, reconnect_cb=reconnect)
    if err_j != 0:
        print(f"[FAIL] GetActualJointPosDegree err={err_j}")
        return err_j

    j_now6 = ensure_joint6(cur_joint)
    j_tgt = list(j_now6)

    # ✅ 0-index: J1~J6 => [0..5], J4 => index 3
    j_tgt[3] = float(j_tgt[3]) + float(delta_deg)

    print(f"[MOVEJ-J4] cur  joint: {fmt_joint(j_now6)}")
    print(f"[MOVEJ-J4] tgt  joint: {fmt_joint(j_tgt)}")
    print(f"[MOVEJ-J4] J4: {j_now6[3]:.3f} -> {j_tgt[3]:.3f} (delta {float(delta_deg):+.3f} deg)")

    rtn = safe_call(
        robot.MoveJ,
        joint_pos=j_tgt,
        tool=int(tool),
        user=int(user),
        vel=float(vel),
        blendT=float(blendT),
        retry=1,
        reconnect_cb=reconnect,
    )
    print(f"[RET] MoveJ(J4) errcode: {rtn}")
    return rtn


def rotate_j4_from_input(
    robot,
    delta_deg: float,
    tool=0,
    user=0,
    reconnect=None,
    vel: float | None = None,
    blendT: float | None = None,
    max_step_deg: float | None = None,
):
    """
    사용자가 입력한 delta_deg로 J4만 회전.
    반환: (ok:bool, delta_deg:float, errcode:int)
    """
    print("\n[ACTION] Rotate J4 by user input")
    print(f"  delta(J4)={float(delta_deg):+.3f} deg")

    err = rotate_j4_by_delta(
        robot=robot,
        delta_deg=float(delta_deg),
        tool=tool,
        user=user,
        vel=vel,
        blendT=blendT,
        reconnect=reconnect,
        max_step_deg=max_step_deg,
    )
    return (err == 0), float(delta_deg), err