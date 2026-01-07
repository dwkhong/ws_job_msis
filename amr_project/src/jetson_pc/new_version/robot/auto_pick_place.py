# robot/auto_pick_place.py
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from . import robot_state as rs
from . import target_pose as tp
from . import ik_check as ik
from . import smooth_auto as sa
from . import return_home as rh
from . import gripper_control as gc
from . import robot_config as rc

# stack place
from . import stack_cycle_11 as sc

# vision
from vision import measure_box as mb


def _need_camera_running() -> bool:
    # mb.is_running()이 있으면 쓰고, 없으면 "그냥 켜져있다고 가정" (호환용)
    if hasattr(mb, "is_running"):
        try:
            return bool(mb.is_running())
        except Exception:
            return True
    return True


def cmd12(robot, reconnect=None) -> Dict[str, Any]:
    """
    ✅ main 12번용:
      N 입력받고
      (3->4->5->6) pick 수행
      HOME(그리퍼 유지)
      place(cmd11_stack_cycle) 수행 (DROP에서 OPEN, 끝에 HOME)
      반복
    """
    if robot is None:
        print("[12] Robot not connected. (0번 먼저)")
        return {"ok": False, "msg": "robot is None"}

    if not _need_camera_running():
        print("[12] 카메라가 꺼져있습니다. (1번으로 ON)")
        return {"ok": False, "msg": "camera not running"}

    raw = input("박스 몇 개 옮길까요? (예: 4, b=back) > ").strip().lower()
    if raw in ("b", "back", "q", "quit"):
        return {"ok": True, "msg": "cancel"}

    try:
        n = int(raw)
        if n <= 0:
            raise ValueError()
    except Exception:
        print("[12] 숫자 입력이 아닙니다.")
        return {"ok": False, "msg": "invalid count"}

    # state는 gripper/stack_counter 공유 (gc가 내부 dict 관리)
    state = gc.get_state()
    state.setdefault("stack_counter", 0)

    tool = int(getattr(rc, "TOOL_ID", 0))
    user = int(getattr(rc, "USER_ID", 0))

    print(f"\n[12] Auto Pick&Place start: {n} cycles (stack_counter={state.get('stack_counter', 0)})")

    for i in range(n):
        print("\n" + "=" * 60)
        print(f"[12] Cycle {i+1}/{n}")
        print("=" * 60)

        # (선택) 시작은 HOME으로 정렬(그리퍼 유지)
        out_home0 = rh.cmd_home_only(robot, reconnect=reconnect)
        if not out_home0.get("ok", False):
            print("[12] HOME(prepare) FAIL:", out_home0.get("msg", ""))
            return {"ok": False, "msg": "home prepare fail", "cycle": i+1}

        # 3) Measure (cache)
        print("[12] Step3: measure_avg")
        mb.cmd_measure_avg()  # 내부 캐시에 last_measure 저장한다고 가정
        time.sleep(0.05)

        # 4) Build target from last
        print("[12] Step4: build target from last")
        tp.cmd_build_target_from_last(robot, reconnect=reconnect, use_last_pose=True)

        # target이 캐시에 들어갔는지 확인 (함수 있으면)
        if hasattr(tp, "get_last_target_pose6"):
            if tp.get_last_target_pose6() is None:
                print("[12] target 생성 실패(캐시 없음)")
                return {"ok": False, "msg": "target cache missing", "cycle": i+1}

        # 5) IK check (cache: phase0)
        print("[12] Step5: IK check from last")
        ik.cmd_check_target_from_last(robot, reconnect=reconnect)

        if hasattr(ik, "get_last_phase0_pose6"):
            if ik.get_last_phase0_pose6() is None:
                print("[12] IK 체크 실패(phase0 캐시 없음)")
                return {"ok": False, "msg": "phase0 cache missing", "cycle": i+1}

        # 6) Smooth pick (uses cached target/phase0)
        print("[12] Step6: smooth pick (cmd6)")
        out_pick = sa.cmd6(robot, reconnect=reconnect)
        if not out_pick.get("ok", False):
            print("[12] PICK FAIL:", out_pick.get("msg", ""))
            return {"ok": False, "msg": "pick fail", "cycle": i+1, "detail": out_pick}

        # ✅ 픽업 후 HOME(그리퍼 유지) — cmd8 쓰면 OPEN돼서 박스 떨어짐!
        print("[12] Step7: HOME only (carry box)")
        out_home1 = rh.cmd_home_only(robot, reconnect=reconnect)
        if not out_home1.get("ok", False):
            print("[12] HOME(carry) FAIL:", out_home1.get("msg", ""))
            return {"ok": False, "msg": "home carry fail", "cycle": i+1}

        # Place: A -> DROP -> OPEN -> A -> HOME (stack_counter++)
        print("[12] Step8: PLACE (stack_cycle_11)")
        home_joint6 = rs.get_initial_joint6() if hasattr(rs, "get_initial_joint6") else None
        out_place = sc.cmd11_stack_cycle(
            robot=robot,
            reconnect=reconnect,
            state=state,
            home_joint6=home_joint6,
            tool=tool,
            user=user,
        )
        if not out_place.get("ok", False):
            print("[12] PLACE FAIL:", out_place.get("msg", ""))
            return {"ok": False, "msg": "place fail", "cycle": i+1, "detail": out_place}

        print(f"[12] ✅ Cycle {i+1} done. stack_counter={state.get('stack_counter')}")

    print("\n[12] 🎉 All cycles done.")
    return {"ok": True, "msg": "done", "count": n, "stack_counter": state.get("stack_counter", 0)}
