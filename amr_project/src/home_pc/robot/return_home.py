# robot/return_home.py
"""
Return Home 클래스
- 초기 위치(Home)로 복귀
- 그리퍼 열기
"""
from __future__ import annotations
from typing import Dict, List, Optional, Callable


class ReturnHome:
    """
    Return Home 클래스
    - 초기 joint 위치로 복귀
    - 그리퍼 자동 열기 옵션
    """
    
    def __init__(self, robot_connector, robot_state, robot_motion, gripper_controller):
        """
        Args:
            robot_connector: RobotConnector 인스턴스
            robot_state: RobotState 인스턴스
            robot_motion: RobotMotion 인스턴스
            gripper_controller: GripperController 인스턴스
        """
        self.connector = robot_connector
        self.robot_state = robot_state
        self.robot_motion = robot_motion
        self.gripper = gripper_controller
    
    # -------------------------
    # Core: 초기 위치로 복귀
    # -------------------------
    def return_to_initial(self, initial_joint6: Optional[List[float]] = None,
                         reconnect_cb: Optional[Callable] = None,
                         reset_after: bool = True) -> Dict[str, object]:
        """
        초기 joint 위치로 복귀
        
        Args:
            initial_joint6: 초기 joint (None이면 자동으로 가져옴)
            reconnect_cb: 재연결 콜백
            reset_after: 복귀 후 상태 리셋 여부
        
        Returns:
            결과 딕셔너리
        """
        robot = self.connector.get_robot()
        if robot is None:
            return {"ok": False, "msg": "robot is None", "err": -1, "reset": False}
        
        # Initial joint 가져오기
        if initial_joint6 is None:
            initial_joint6 = self.robot_state.get_initial_joint6()
            
            # 없으면 현재 위치를 초기 위치로 저장
            if initial_joint6 is None:
                self.robot_state.try_capture_initial_joint(
                    reconnect_cb=reconnect_cb,
                    verbose=False
                )
                initial_joint6 = self.robot_state.get_initial_joint6()
        
        if initial_joint6 is None:
            return {"ok": False, "msg": "초기 joint가 저장되어 있지 않습니다.", "err": -1, "reset": False}
        
        # 현재 joint 출력
        try:
            err_j, cur_joint = self.robot_state.read_joint6(reconnect_cb=reconnect_cb)
            if err_j == 0 and cur_joint is not None:
                dlt = [float(a) - float(b) for a, b in zip(initial_joint6[:6], cur_joint[:6])]
                print(f"[HOME] current: {self.robot_state.fmt_joint(cur_joint)}")
                print(f"[HOME] target : {self.robot_state.fmt_joint(initial_joint6)}")
                print(f"[HOME] delta  : [" + ", ".join(f"{v:+.3f}" for v in dlt) + "]")
        except Exception:
            pass
        
        # MoveJ로 복귀
        print("\n[ACTION] HOME 복귀 (MoveJ)")
        err = self.robot_motion.move_j(
            joint_pos=initial_joint6,
            label="HOME",
            reconnect_cb=reconnect_cb
        )
        
        ok = (err == 0)
        return {
            "ok": ok,
            "err": err,
            "reset": bool(reset_after),
            "msg": "HOME 복귀 완료" if ok else f"HOME 복귀 실패 errcode={err}"
        }
    
    # -------------------------
    # 명령 (8번 기능)
    # -------------------------
    def cmd8_return_home_and_open(self, reconnect_cb: Optional[Callable] = None) -> Dict[str, object]:
        """
        8번: Home 복귀 + 그리퍼 열기
        
        Args:
            reconnect_cb: 재연결 콜백
        
        Returns:
            결과 딕셔너리
        """
        if not self.connector.is_connected():
            print("[8] 로봇이 연결되어 있지 않습니다. (0번으로 연결)")
            return {"ok": False, "msg": "robot is None", "err": -1, "reset": False}
        
        # Home 복귀
        out = self.return_to_initial(reconnect_cb=reconnect_cb, reset_after=True)
        
        # 성공하면 그리퍼 열기
        if out.get("ok", False):
            print("[ACTION] HOME 도착 → Gripper OPEN")
            self.gripper.open(reconnect_cb=reconnect_cb)
        
        return out
    
    # -------------------------
    # Home만 복귀 (그리퍼 유지)
    # -------------------------
    def cmd_home_only(self, reconnect_cb: Optional[Callable] = None) -> Dict[str, object]:
        """
        Home만 복귀 (그리퍼 상태 유지)
        12번 루프용
        
        Args:
            reconnect_cb: 재연결 콜백
        
        Returns:
            결과 딕셔너리
        """
        if not self.connector.is_connected():
            return {"ok": False, "msg": "robot is None", "err": -1}
        
        return self.return_to_initial(reconnect_cb=reconnect_cb, reset_after=False)