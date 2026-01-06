# robot/j6_rotate.py
import time

# ✅ robot_config를 단일 소스로 사용
# (패키지/단일 실행 모두 대비)
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
# Core
# -------------------------
def calc_delta_from_measure(
    last_measure: dict,
    angle_key: str | None = None,
    sign: float | None = None,
    max_step_deg: float | None = None,
) -> float:
    """
    비전 측정 결과(last_measure)에서 angle 값을 읽어서
    J6 회전 delta(deg)를 만들어줌 (clamp 포함)

    기본값은 robot_config(rc) 사용:
      - angle_key     = rc.ANGLE_KEY
      - sign          = rc.ANGLE_TO_J6_SIGN
      - max_step_deg  = rc.J6_MAX_STEP_DEG
    """
    if last_measure is None:
        raise ValueError("last_measure is None")

    if angle_key is None:
        angle_key = getattr(rc, "ANGLE_KEY", "angle_deg")
    if sign is None:
        sign = float(getattr(rc, "ANGLE_TO_J6_SIGN", +1.0))
    if max_step_deg is None:
        max_step_deg = float(getattr(rc, "J6_MAX_STEP_DEG", 45.0))

    if angle_key not in last_measure:
        raise KeyError(f"'{angle_key}' not found in last_measure keys={list(last_measure.keys())}")

    ang = float(last_measure.get(angle_key, 0.0))
    delta = float(sign) * ang
    delta = clamp(delta, -float(max_step_deg), +float(max_step_deg))
    return delta


def rotate_j6_by_delta(
    robot,
    delta_deg: float,
    tool=0,
    user=0,
    vel: float | None = None,
    blendT: float | None = None,
    reconnect=None,
):
    """
    현재 조인트를 읽어서 J6만 delta_deg만큼 더한 뒤 MoveJ 수행
    반환: errcode(int)

    기본 vel/blendT는 robot_config(rc) 사용:
      - vel    = rc.MOVEJ_VEL_J6
      - blendT = rc.MOVEJ_BLENDT_J6
    """
    if vel is None:
        vel = float(getattr(rc, "MOVEJ_VEL_J6", 100.0))
    if blendT is None:
        blendT = float(getattr(rc, "MOVEJ_BLENDT_J6", -1.0))

    err_j, cur_joint = safe_call(robot.GetActualJointPosDegree, flag=1, retry=1, reconnect_cb=reconnect)
    if err_j != 0:
        print(f"[FAIL] GetActualJointPosDegree err={err_j}")
        return err_j

    j_now6 = ensure_joint6(cur_joint)
    j_tgt = list(j_now6)
    j_tgt[5] = float(j_tgt[5]) + float(delta_deg)

    print(f"[MOVEJ-J6] cur  joint: {fmt_joint(j_now6)}")
    print(f"[MOVEJ-J6] tgt  joint: {fmt_joint(j_tgt)}")
    print(f"[MOVEJ-J6] J6: {j_now6[5]:.3f} -> {j_tgt[5]:.3f} (delta {float(delta_deg):+.3f} deg)")

    rtn = safe_call(
        robot.MoveJ,
        joint_pos=j_tgt,
        tool=int(tool),
        user=int(user),
        vel=float(vel),
        blendT=float(blendT),
        retry=1,
        reconnect_cb=reconnect
    )
    print(f"[RET] MoveJ(J6) errcode: {rtn}")
    return rtn


def rotate_j6_from_measure(
    robot,
    last_measure: dict,
    tool=0,
    user=0,
    reconnect=None,
    # ✅ 필요하면 override 가능 (기본은 rc)
    angle_key: str | None = None,
    sign: float | None = None,
    max_step_deg: float | None = None,
    vel: float | None = None,
    blendT: float | None = None,
):
    """
    last_measure[angle_key] -> delta -> rotate J6
    반환: (ok:bool, delta_deg:float, errcode:int)

    기본 설정은 robot_config(rc)에서 읽음.
    """
    if angle_key is None:
        angle_key = getattr(rc, "ANGLE_KEY", "angle_deg")

    delta = calc_delta_from_measure(
        last_measure=last_measure,
        angle_key=angle_key,
        sign=sign,
        max_step_deg=max_step_deg,
    )

    print("\n[ACTION] Rotate J6 by measured angle")
    print(f"  {angle_key}={float(last_measure.get(angle_key, 0.0)):+.3f} => delta(J6)={delta:+.3f} deg")

    err = rotate_j6_by_delta(
        robot=robot,
        delta_deg=delta,
        tool=tool,
        user=user,
        vel=vel,
        blendT=blendT,
        reconnect=reconnect,
    )
    return (err == 0), delta, err
