# robot/auto_pick_place.py
"""
자동 Pick & Place 클래스
- N번 반복 자동 실행
- Pick → Home → Place(스택) 사이클
"""
from __future__ import annotations
import time
from typing import Any, Dict, Optional, Callable

from config import robot_config as cfg


class AutoPickPlace:
    """
    자동 Pick & Place 클래스
    - 측정 → 목표 계산 → IK 검증 → Pick → Place 자동 반복
    - 스택 카운터 관리
    - Vision restart 제어 (안정화)
    """
    
    def __init__(self, robot_connector, robot_state, robot_motion,
                 gripper_controller, box_detector, target_pose,
                 ik_checker, pick_place, return_home):
        """
        Args:
            robot_connector: RobotConnector 인스턴스
            robot_state: RobotState 인스턴스
            robot_motion: RobotMotion 인스턴스
            gripper_controller: GripperController 인스턴스
            box_detector: BoxDetector 인스턴스
            target_pose: TargetPose 인스턴스
            ik_checker: IKChecker 인스턴스
            pick_place: PickPlace 인스턴스
            return_home: ReturnHome 인스턴스
        """
        self.connector = robot_connector
        self.robot_state = robot_state
        self.robot_motion = robot_motion
        self.gripper = gripper_controller
        self.box_detector = box_detector
        self.target_pose = target_pose
        self.ik_checker = ik_checker
        self.pick_place = pick_place
        self.return_home = return_home
        
        # 스택 카운터
        self._stack_counter = 0
    
    # -------------------------
    # 유틸리티
    # -------------------------
    def get_stack_counter(self) -> int:
        """스택 카운터 반환"""
        return self._stack_counter
    
    def reset_stack_counter(self):
        """스택 카운터 초기화"""
        self._stack_counter = 0
    
    def set_vision_allow_restart(self, allow: bool):
        """
        Vision restart 허용/금지 (안정화용)
        - 측정 중: allow=True (restart 허용)
        - 로봇 동작 중: allow=False (restart 금지, 크래시 방지)
        """
        if hasattr(self.box_detector, 'set_allow_restart'):
            try:
                self.box_detector.set_allow_restart(bool(allow))
            except Exception:
                pass
    
    def get_home_joint6(self, reconnect_cb: Optional[Callable] = None):
        """
        Home joint 가져오기
        1) initial_joint6 (초기 저장값)
        2) last_joint6 (2번에서 읽은 값)
        3) 현재 joint 읽기 (최후 수단)
        """
        # 1) initial
        hj = self.robot_state.get_initial_joint6()
        if hj is not None:
            return hj
        
        # 2) last
        hj = self.robot_state.get_last_joint6()
        if hj is not None:
            return hj
        
        # 3) 현재 읽기
        try:
            (e1, _), (e2, j) = self.robot_state.read_pose_joint(reconnect_cb=reconnect_cb)
            if e2 == 0 and j is not None:
                return j
        except Exception:
            pass
        
        return None
    
    # -------------------------
    # Place (스택)
    # -------------------------
    def place_one_stack(self, home_joint6, reconnect_cb: Optional[Callable] = None) -> Dict[str, Any]:
        """
        스택 위치에 박스 놓기
        A(MoveJ) -> DROP(MoveCart) -> Gripper OPEN -> A(MoveJ) -> HOME(MoveJ) -> counter++
        
        Args:
            home_joint6: Home joint 위치
            reconnect_cb: 재연결 콜백
        
        Returns:
            결과 딕셔너리
        """
        if home_joint6 is None:
            return {"ok": False, "msg": "home_joint6 없음 (2번으로 홈 위치 저장 필요)"}
        
        # Drop 위치 계산 (스택 높이 반영)
        drop = list(cfg.WP11_DROP_BASE_POSE)
        drop[2] = float(drop[2]) + float(cfg.STACK_Z_STEP_MM) * self._stack_counter
        
        # 높이 제한 체크
        if cfg.STACK_Z_MAX_MM is not None and float(drop[2]) > float(cfg.STACK_Z_MAX_MM):
            return {
                "ok": False,
                "msg": f"DROP Z too high: {drop[2]:.1f} > {float(cfg.STACK_Z_MAX_MM):.1f}",
                "drop": drop
            }
        
        print(f"\n[PLACE] counter={self._stack_counter} dropZ={drop[2]:.1f}")
        
        # 1) A 위치로 이동
        r = self.robot_motion.move_j(
            joint_pos=cfg.WP11_A_JOINT,
            vel=cfg.MOVEJ_VEL_WP11,
            label="A",
            reconnect_cb=reconnect_cb
        )
        if int(r) != 0:
            return {"ok": False, "msg": f"MoveJ(A) err={r}"}
        
        # 2) DROP 위치로 이동
        r = self.robot_motion.move_cart(
            pose6=drop,
            label="DROP",
            reconnect_cb=reconnect_cb
        )
        if int(r) != 0:
            return {"ok": False, "msg": f"MoveCart(DROP) err={r}", "drop": drop}
        
        # 3) 그리퍼 열기 (놓기)
        try:
            self.gripper.open(reconnect_cb=reconnect_cb)
        except Exception as e:
            return {"ok": False, "msg": f"gripper_open failed: {e}", "drop": drop}
        
        # 4) A 위치로 복귀
        r = self.robot_motion.move_j(
            joint_pos=cfg.WP11_A_JOINT,
            vel=cfg.MOVEJ_VEL_WP11,
            label="A back",
            reconnect_cb=reconnect_cb
        )
        if int(r) != 0:
            return {"ok": False, "msg": f"MoveJ(A back) err={r}"}
        
        # 5) HOME으로 복귀
        r = self.robot_motion.move_j(
            joint_pos=home_joint6,
            vel=cfg.MOVEJ_VEL_RETURN,
            label="HOME",
            reconnect_cb=reconnect_cb
        )
        if int(r) != 0:
            return {"ok": False, "msg": f"MoveJ(HOME) err={r}"}
        
        # 카운터 증가
        self._stack_counter += 1
        
        return {
            "ok": True,
            "msg": f"PLACE done. counter->{self._stack_counter}",
            "drop": drop
        }
    
    # -------------------------
    # 명령 (12번 기능)
    # -------------------------
    def cmd12_auto_loop(self, reconnect_cb: Optional[Callable] = None) -> Dict[str, Any]:
        """
        12번: 자동 Pick & Place 루프
        N번 반복: 측정 → 계산 → IK → Pick → Home → Place
        
        안정화:
        - Step3(측정) 중에는 vision restart 허용
        - Step4~8(로봇 동작) 중에는 vision restart 금지 (크래시 방지)
        - 종료/에러/리턴 시 반드시 restart 허용으로 복구
        
        Args:
            reconnect_cb: 재연결 콜백
        
        Returns:
            결과 딕셔너리
        """
        if not self.connector.is_connected():
            print("[12] 로봇이 연결되어 있지 않습니다. (0번으로 연결)")
            return {"ok": False, "msg": "robot is None"}
        
        if not self.box_detector.is_running():
            print("[12] 카메라가 꺼져있습니다. (1번으로 ON)")
            return {"ok": False, "msg": "camera not running"}
        
        # 반복 횟수 입력
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
        
        # Home joint 가져오기
        home_joint6 = self.get_home_joint6(reconnect_cb=reconnect_cb)
        if home_joint6 is None:
            print("[12] home_joint6를 못 찾았음. 2번(홈 저장)부터 하세요.")
            return {"ok": False, "msg": "home_joint6 missing"}
        
        print(f"\n[12] Auto Pick&Place start: {n} cycles (stack_counter={self._stack_counter})")
        
        # 어떤 이유로든 cmd12가 끝나면 restart 허용으로 복구
        self.set_vision_allow_restart(True)
        
        try:
            for i in range(n):
                print("\n" + "=" * 60)
                print(f"[12] Cycle {i+1}/{n}")
                print("=" * 60)
                
                # 0) HOME 정렬 (그리퍼 유지)
                out_home0 = self.return_home.cmd_home_only(reconnect_cb=reconnect_cb)
                if not out_home0.get("ok", False):
                    print("[12] HOME(prepare) FAIL:", out_home0.get("msg", ""))
                    return {"ok": False, "msg": "home prepare fail", "cycle": i+1}
                
                # HOME 후 pose/joint 캐시 갱신
                try:
                    self.robot_state.read_pose_joint(reconnect_cb=reconnect_cb)
                except Exception:
                    pass
                
                # 3) Measure (측정 중에는 restart 허용)
                print("[12] Step3: measure_avg")
                self.set_vision_allow_restart(True)
                meas = self.box_detector.cmd_measure_avg()
                time.sleep(0.05)
                
                if meas is None:
                    print("[12] measure_avg 실패")
                    return {"ok": False, "msg": "measure fail", "cycle": i+1}
                
                # 이제부터 로봇이 움직일 거라 restart 금지
                self.set_vision_allow_restart(False)
                
                # 4) Build target
                print("[12] Step4: build target from last")
                self.target_pose.cmd_build_target_from_last(
                    use_last_pose=True,
                    reconnect_cb=reconnect_cb
                )
                
                if self.target_pose.get_last_target_pose6() is None:
                    print("[12] target 생성 실패(캐시 없음)")
                    return {"ok": False, "msg": "target cache missing", "cycle": i+1}
                
                # 5) IK check
                print("[12] Step5: IK check from last")
                self.ik_checker.cmd_check_target_from_last(
                    check_phase0=True,
                    reconnect_cb=reconnect_cb
                )
                
                if self.ik_checker.get_last_phase0_pose6() is None:
                    print("[12] IK 체크 실패(phase0 캐시 없음)")
                    return {"ok": False, "msg": "phase0 cache missing", "cycle": i+1}
                
                # 6) Smooth pick
                print("[12] Step6: smooth pick (cmd6)")
                out_pick = self.pick_place.cmd6_smooth_auto(reconnect_cb=reconnect_cb)
                if not out_pick.get("ok", False):
                    print("[12] PICK FAIL:", out_pick.get("msg", ""))
                    return {"ok": False, "msg": "pick fail", "cycle": i+1, "detail": out_pick}
                
                # 7) HOME only (carry)
                print("[12] Step7: HOME only (carry box)")
                out_home1 = self.return_home.cmd_home_only(reconnect_cb=reconnect_cb)
                if not out_home1.get("ok", False):
                    print("[12] HOME(carry) FAIL:", out_home1.get("msg", ""))
                    return {"ok": False, "msg": "home carry fail", "cycle": i+1}
                
                # 8) PLACE
                print("[12] Step8: PLACE (A->DROP->OPEN->A->HOME)")
                out_place = self.place_one_stack(
                    home_joint6=home_joint6,
                    reconnect_cb=reconnect_cb
                )
                if not out_place.get("ok", False):
                    print("[12] PLACE FAIL:", out_place.get("msg", ""))
                    return {"ok": False, "msg": "place fail", "cycle": i+1, "detail": out_place}
                
                print(f"[12] ✅ Cycle {i+1} done. stack_counter={self._stack_counter}")
                
                # 다음 사이클 측정 전에 restart 허용으로 풀어둠 (안전)
                self.set_vision_allow_restart(True)
            
            print("\n[12] 🎉 All cycles done.")
            return {
                "ok": True,
                "msg": "done",
                "count": n,
                "stack_counter": self._stack_counter
            }
        
        except Exception as e:
            print(f"[12] Exception: {e}")
            return {"ok": False, "msg": f"exception: {e}"}
        
        finally:
            # 어떤 종료 경로든 vision restart 허용 복구
            self.set_vision_allow_restart(True)