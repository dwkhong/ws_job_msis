# robot/j4_rotate.py
from __future__ import annotations

import time
from typing import Optional, Dict, Any

# ✅ robot_config를 단일 소스로 사용 (패키지/단일 실행 모두 대비)
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
# Core: "입력값(delta deg)만큼" J4 이동
# -------------------------
def rotate_j4_by_delta(
    robot,
    delta_deg: float,
    tool: Optional[int] = None,
    user: Optional[int] = None,
    vel: Optional[float] = None,
    blendT: Optional[float] = None,
    reconnect=None,
) -> int:
    """
    현재 조인트를 읽어서 J4만 delta_deg만큼 더한 뒤 MoveJ 수행
    반환: errcode(int)

    기본값(robot_config):
      - tool   = rc.TOOL_ID
      - user   = rc.USER_ID
      - vel    = rc.MOVEJ_VEL_J4
      - blendT = rc.MOVEJ_BLENDT_J4
      - (옵션) delta clamp: rc.J4_MAX_STEP_DEG
    """
    if tool is None:
        tool = int(getattr(rc, "TOOL_ID", 0))
    if user is None:
        user = int(getattr(rc, "USER_ID", 0))
    if vel is None:
        vel = float(getattr(rc, "MOVEJ_VEL_J4", 60.0))
    if blendT is None:
        blendT = float(getattr(rc, "MOVEJ_BLENDT_J4", -1.0))

    # ✅ (선택) 한 번에 너무 크게 못 움직이게 clamp
    max_step = float(getattr(rc, "J4_MAX_STEP_DEG", 9999.0))
    delta_deg = clamp(float(delta_deg), -max_step, +max_step)

    err_j, cur_joint = safe_call(
        robot.GetActualJointPosDegree,
        flag=1,
        retry=int(getattr(rc, "RPC_RETRY", 1)),
        sleep_sec=float(getattr(rc, "RPC_RETRY_SLEEP_SEC", 0.25)),
        reconnect_cb=reconnect
    )
    if err_j != 0:
        print(f"[FAIL] GetActualJointPosDegree err={err_j}")
        return int(err_j)

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
        retry=int(getattr(rc, "RPC_RETRY", 1)),
        sleep_sec=float(getattr(rc, "RPC_RETRY_SLEEP_SEC", 0.25)),
        reconnect_cb=reconnect
    )
    print(f"[RET] MoveJ(J4) errcode: {rtn}")
    return int(rtn)


def cmd_rotate_j4_prompt(
    robot,
    reconnect=None,
    tool: Optional[int] = None,
    user: Optional[int] = None,
    vel: Optional[float] = None,
    blendT: Optional[float] = None,
) -> Dict[str, Any]:
    """
    ✅ 콘솔 입력으로 J4 delta(deg) 입력받아 회전
    out:
      ok(bool), delta(float), err(int), msg(str)
    """
    out: Dict[str, Any] = {"ok": False, "delta": 0.0, "err": -1, "msg": ""}

    s = input("J4 delta(deg) 입력 (예: 5 / -3.2, b=back) > ").strip().lower()
    if s in ("b", "back", "q", "quit"):
        out["msg"] = "cancel"
        out["err"] = 0
        return out

    try:
        delta = float(s)
    except Exception:
        out["msg"] = f"invalid input: {s}"
        out["err"] = -2
        return out

    print("\n[ACTION] Rotate J4 by user input delta")
    print(f"  delta(J4) = {delta:+.3f} deg")

    err = rotate_j4_by_delta(
        robot=robot,
        delta_deg=delta,
        tool=tool,
        user=user,
        vel=vel,
        blendT=blendT,
        reconnect=reconnect
    )

    out["delta"] = float(delta)
    out["err"] = int(err)
    out["ok"] = (err == 0)
    out["msg"] = "ok" if out["ok"] else f"MoveJ failed errcode={err}"
    return out


def cmd10(robot, reconnect=None) -> Dict[str, Any]:
    """
    ✅ main의 10번용: J4 delta 입력받아 회전
    """
    if robot is None:
        print("[10] Robot not connected. (0번 먼저)")
        return {"ok": False, "msg": "robot is None", "err": -1, "delta": 0.0}

    return cmd_rotate_j4_prompt(robot, reconnect=reconnect)