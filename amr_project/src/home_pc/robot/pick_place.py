# robot/pick_place.py
"""
Pick & Place 워크플로우 클래스
- 6번: 1-step (Phase0 → Target, 한 번에)
- 7번: 2-step (Phase0 → 대기 → Target, 두 번 누름)
"""
from __future__ import annotations
import time
from typing import Dict, Any, Optional, List, Callable

from config import robot_config as cfg


class PickPlace:
    """
    Pick & Place 워크플로우 클래스
    - Smooth Auto: Phase0 → Target 자동 이동
    - 2-Step: Phase0까지 이동 대기 → Target으로 하강
    """
    
    def __init__(self, robot_connector, robot_state, robot_motion, 
                 gripper_controller, ik_checker, target_pose):
        """
        Args:
            robot_connector: RobotConnector 인스턴스
            robot_state: RobotState 인스턴스
            robot_motion: RobotMotion 인스턴스
            gripper_controller: GripperController 인스턴스
            ik_checker: IKChecker 인스턴스
            target_pose: TargetPose 인스턴스
        """
        self.connector = robot_connector
        self.robot_state = robot_state
        self.robot_motion = robot_motion
        self.gripper = gripper_controller
        self.ik_checker = ik_checker
        self.target_pose = target_pose
        
        # 7번용 2-step 컨텍스트
        self._cmd7_ctx: Dict[str, Any] = {
            "armed": False,
            "target": None,
            "phase0": None,
            "t": 0.0,
        }
    
    # -------------------------
    # 유틸리티
    # -------------------------
    @staticmethod
    def ensure_pose6(p) -> List[float]:
        """Pose6 검증"""
        if not isinstance(p, (list, tuple)) or len(p) < 6:
            raise ValueError("pose6 must be list/tuple len>=6")
        return [float(x) for x in p[:6]]
    
    @staticmethod
    def same_pose6(a: Optional[List], b: Optional[List], tol: float = 1e-6) -> bool:
        """두 pose가 같은지 비교"""
        if a is None or b is None:
            return False
        aa = PickPlace.ensure_pose6(a)
        bb = PickPlace.ensure_pose6(b)
        return all(abs(float(x) - float(y)) <= tol for x, y in zip(aa, bb))
    
    def reset_cmd7(self):
        """7번 컨텍스트 초기화"""
        self._cmd7_ctx["armed"] = False
        self._cmd7_ctx["target"] = None
        self._cmd7_ctx["phase0"] = None
        self._cmd7_ctx["t"] = 0.0
    
    # -------------------------
    # Target/Phase0 가져오기
    # -------------------------
    def get_best_target_and_phase0(self) -> tuple:
        """
        최적의 target과 phase0 반환
        - target: IK에서 성공한 ok_target 우선, 없으면 기본 target
        - phase0: IK 결과의 phase0
        
        Returns:
            (target, phase0) 튜플
        """
        # 기본 target
        target = self.target_pose.get_last_target_pose6()
        
        # IK에서 성공한 ok_target이 있으면 우선 사용
        ok_target = self.ik_checker.get_last_ok_target_pose6()
        if ok_target is not None:
            target = ok_target
        
        # Phase0
        phase0 = self.ik_checker.get_last_phase0_pose6()
        
        return target, phase0
    
    # -------------------------
    # Core: Smooth Auto (Phase0 → Target)
    # -------------------------
    def smooth_auto(self, target_pose6, phase0_pose6,
                    auto_grip_close: bool = True,
                    reconnect_cb: Optional[Callable] = None) -> Dict[str, Any]:
        """
        자동 접근 (Phase0 → Target) + 그리퍼 닫기
        
        Args:
            target_pose6: 최종 target pose
            phase0_pose6: 안전 진입점 (Phase0)
            auto_grip_close: 자동 그리퍼 닫기 여부
            reconnect_cb: 재연결 콜백
        
        Returns:
            결과 딕셔너리
        """
        robot = self.connector.get_robot()
        if robot is None:
            return {"ok": False, "msg": "robot is None"}
        
        if target_pose6 is None or phase0_pose6 is None:
            return {"ok": False, "msg": "target/phase0 is None"}
        
        target = self.ensure_pose6(target_pose6)
        phase0 = self.ensure_pose6(phase0_pose6)
        
        # 0) 현재 상태 읽기
        (e1, _), (e2, cur_joint) = self.robot_state.read_pose_joint(reconnect_cb=reconnect_cb)
        if e1 != 0 or e2 != 0 or cur_joint is None:
            return {"ok": False, "msg": f"상태 읽기 실패 err_p={e1}, err_j={e2}"}
        
        # 1) Phase0 IK 체크 + 이동
        if not self.ik_checker.has_solution(phase0, cur_joint, reconnect_cb=reconnect_cb):
            return {"ok": False, "msg": "Phase0 IK 불가", "phase0": phase0}
        
        print("\n[SMOOTH] 1) Phase0 MoveCart")
        r = self.robot_motion.move_cart(
            phase0,
            label="phase0",
            reconnect_cb=reconnect_cb
        )
        if int(r) != 0:
            return {"ok": False, "msg": f"Phase0 이동 실패 err={r}", "phase0": phase0}
        
        # 2) Phase0 도착 후 다시 joint 읽어서 target IK 확인 + 이동
        (ep2, _), (ej2, joint_at_phase0) = self.robot_state.read_pose_joint(reconnect_cb=reconnect_cb)
        if ep2 != 0 or ej2 != 0 or joint_at_phase0 is None:
            return {"ok": False, "msg": f"phase0 이후 상태 읽기 실패 err_p={ep2}, err_j={ej2}"}
        
        if not self.ik_checker.has_solution(target, joint_at_phase0, reconnect_cb=reconnect_cb):
            return {"ok": False, "msg": "Target IK 불가 (하강 경로 막힘)", "target": target}
        
        print("[SMOOTH] 2) Target MoveCart (down) - PRECISE")
        r = self.robot_motion.move_cart(
            target,
            label="target",
            precise=True,  # 정밀 모드: Config의 PRECISE 속도 사용
            reconnect_cb=reconnect_cb
        )
        if int(r) != 0:
            return {"ok": False, "msg": f"Target 이동 실패 err={r}", "target": target}
        
        # 2.5) 그리퍼 동작 전 안정화 대기 (Config)
        import time
        from config import robot_config as cfg
        time.sleep(cfg.GRIPPER_SETTLE_TIME)
        
        # 3) 그리퍼 닫기
        if auto_grip_close:
            print("[SMOOTH] 3) Gripper close")
            self.gripper.close(reconnect_cb=reconnect_cb)
        
        # 4) 완료 상태
        (e_end_p, pose_end), (e_end_j, joint_end) = self.robot_state.read_pose_joint(reconnect_cb=reconnect_cb)
        if e_end_p == 0 and pose_end is not None:
            print("[SMOOTH] done pose:", self.robot_state.fmt_pose6(pose_end))
        
        return {
            "ok": True,
            "msg": "Smooth auto 완료",
            "phase0": phase0,
            "target": target,
            "pose_end": pose_end if e_end_p == 0 else None,
            "joint_end": joint_end if e_end_j == 0 else None,
        }
    
    # -------------------------
    # 명령 (6번: 1-step)
    # -------------------------
    def cmd6_smooth_auto(self, reconnect_cb: Optional[Callable] = None) -> Dict[str, Any]:
        """
        6번: Smooth Auto (한 번에 Phase0 → Target)
        
        Args:
            reconnect_cb: 재연결 콜백
        
        Returns:
            결과 딕셔너리
        """
        if not self.connector.is_connected():
            print("[6] 로봇이 연결되어 있지 않습니다. (0번으로 연결)")
            return {"ok": False, "msg": "robot is None"}
        
        target, phase0 = self.get_best_target_and_phase0()
        
        if target is None:
            print("[6] last target 없음 (4번 먼저 실행)")
            return {"ok": False, "msg": "last target is None"}
        
        if phase0 is None:
            print("[6] last phase0 없음 (5번에서 phase0 OK 먼저)")
            return {"ok": False, "msg": "last phase0 is None"}
        
        auto_grip_close = bool(getattr(cfg, "AUTO_GRIP_CLOSE", True))
        
        out = self.smooth_auto(
            target_pose6=target,
            phase0_pose6=phase0,
            auto_grip_close=auto_grip_close,
            reconnect_cb=reconnect_cb
        )
        
        if not out.get("ok"):
            print("[6] FAIL:", out.get("msg"))
        
        return out
    
    # -------------------------
    # 명령 (7번: 2-step)
    # -------------------------
    def cmd7_two_step(self, reconnect_cb: Optional[Callable] = None) -> Dict[str, Any]:
        """
        7번: 2-step (Phase0까지 → 대기 → Target으로 하강)
        - 1회: Phase0까지만 이동하고 armed=True
        - 2회: Target으로 내려가고 그리퍼 닫기
        
        Args:
            reconnect_cb: 재연결 콜백
        
        Returns:
            결과 딕셔너리
        """
        if not self.connector.is_connected():
            print("[7] 로봇이 연결되어 있지 않습니다. (0번으로 연결)")
            self.reset_cmd7()
            return {"ok": False, "msg": "robot is None"}
        
        target, phase0 = self.get_best_target_and_phase0()
        
        if target is None:
            print("[7] last target 없음 (4번 먼저 실행)")
            self.reset_cmd7()
            return {"ok": False, "msg": "last target is None"}
        
        if phase0 is None:
            print("[7] last phase0 없음 (5번에서 phase0 OK 먼저)")
            self.reset_cmd7()
            return {"ok": False, "msg": "last phase0 is None"}
        
        target = self.ensure_pose6(target)
        phase0 = self.ensure_pose6(phase0)
        
        # 캐시가 바뀌었으면 리셋
        armed = bool(self._cmd7_ctx.get("armed", False))
        if armed and (not self.same_pose6(self._cmd7_ctx.get("target"), target) or 
                      not self.same_pose6(self._cmd7_ctx.get("phase0"), phase0)):
            self.reset_cmd7()
            armed = False
        
        auto_grip_close = bool(getattr(cfg, "AUTO_GRIP_CLOSE", True))
        
        # ========================================
        # 1회차: Phase0까지만 이동
        # ========================================
        if not armed:
            (e1, _), (e2, cur_joint) = self.robot_state.read_pose_joint(reconnect_cb=reconnect_cb)
            if e1 != 0 or e2 != 0 or cur_joint is None:
                return {"ok": False, "msg": f"상태 읽기 실패 err_p={e1}, err_j={e2}"}
            
            if not self.ik_checker.has_solution(phase0, cur_joint, reconnect_cb=reconnect_cb):
                return {"ok": False, "msg": "Phase0 IK 불가", "phase0": phase0}
            
            print("\n[CMD7] 1) Phase0 MoveCart (ONLY)")
            r = self.robot_motion.move_cart(
                phase0,
                label="phase0",
                reconnect_cb=reconnect_cb
            )
            if int(r) != 0:
                return {"ok": False, "msg": f"Phase0 이동 실패 err={r}", "phase0": phase0}
            
            # Armed 상태 설정
            self._cmd7_ctx["armed"] = True
            self._cmd7_ctx["target"] = target
            self._cmd7_ctx["phase0"] = phase0
            self._cmd7_ctx["t"] = time.time()
            
            print("[CMD7] Phase0 도착. 7번 한 번 더 누르면 내려가서 집습니다.")
            return {"ok": True, "msg": "phase0 reached (press 7 again)", "armed": True}
        
        # ========================================
        # 2회차: Target으로 내려가고 그리퍼 닫기
        # ========================================
        (e1, _), (e2, joint_at_phase0) = self.robot_state.read_pose_joint(reconnect_cb=reconnect_cb)
        if e1 != 0 or e2 != 0 or joint_at_phase0 is None:
            return {"ok": False, "msg": f"phase0 이후 상태 읽기 실패 err_p={e1}, err_j={e2}"}
        
        if not self.ik_checker.has_solution(target, joint_at_phase0, reconnect_cb=reconnect_cb):
            self.reset_cmd7()
            return {"ok": False, "msg": "Target IK 불가 (하강 경로 막힘)", "target": target}
        
        print("\n[CMD7] 2) Target MoveCart (DOWN)")
        r = self.robot_motion.move_cart(
            target,
            label="target",
            reconnect_cb=reconnect_cb
        )
        if int(r) != 0:
            self.reset_cmd7()
            return {"ok": False, "msg": f"Target 이동 실패 err={r}", "target": target}
        
        # 그리퍼 닫기
        if auto_grip_close:
            print("[CMD7] 3) Gripper close")
            self.gripper.close(reconnect_cb=reconnect_cb)
        
        # 완료
        (e_end_p, pose_end), _ = self.robot_state.read_pose_joint(reconnect_cb=reconnect_cb)
        if e_end_p == 0 and pose_end is not None:
            print("[CMD7] done pose:", self.robot_state.fmt_pose6(pose_end))
        
        self.reset_cmd7()
        return {
            "ok": True,
            "msg": "descend+grip done",
            "armed": False,
            "pose_end": pose_end if e_end_p == 0 else None
        }
