# robot/gripper_controller.py
"""
그리퍼 제어 클래스
- 그리퍼 활성화
- Open/Close/Toggle
- 상태 관리
"""
from __future__ import annotations
import time
from typing import Dict, Any, Optional, Callable

from config import robot_config as cfg


class GripperController:
    """
    그리퍼 제어 클래스
    - 그리퍼 활성화 및 제어
    - 상태 추적 (activated, closed)
    """
    
    def __init__(self, robot_connector, robot_state):
        """
        Args:
            robot_connector: RobotConnector 인스턴스
            robot_state: RobotState 인스턴스
        """
        self.connector = robot_connector
        self.robot_state = robot_state
        
        # 그리퍼 상태
        self._state: Dict[str, Any] = {
            "gripper_activated": False,
            "gripper_closed": None,  # True/False/None(unknown)
        }
    
    # -------------------------
    # 상태 관리
    # -------------------------
    def get_state(self) -> Dict[str, Any]:
        """그리퍼 상태 반환"""
        return self._state
    
    def reset_state(self):
        """그리퍼 상태 초기화"""
        self._state["gripper_activated"] = False
        self._state["gripper_closed"] = None
    
    # -------------------------
    # 그리퍼 활성화
    # -------------------------
    def ensure_activated(self, reconnect_cb: Optional[Callable] = None) -> bool:
        """
        그리퍼 활성화 (필요 시)
        
        Args:
            reconnect_cb: 재연결 콜백
        
        Returns:
            bool: 활성화 성공 시 True
        """
        if self._state.get("gripper_activated", False):
            return True
        
        robot = self.connector.get_robot()
        if robot is None:
            return False
        
        print("[GRIP] Activating gripper...")
        try:
            err = self.robot_state.safe_call(
                robot.ActGripper,
                cfg.GRIPPER_INDEX,
                1,
                reconnect_cb=reconnect_cb
            )
        except Exception as e:
            print(f"[GRIP-FAIL] ActGripper exception: {e}")
            return False
        
        print(f"[GRIP] ActGripper: err={err}")
        if int(err) == 0:
            self._state["gripper_activated"] = True
        
        time.sleep(0.3)
        return int(err) == 0
    
    # -------------------------
    # 그리퍼 이동
    # -------------------------
    def move(self, pos: int, reconnect_cb: Optional[Callable] = None) -> int:
        """
        그리퍼를 지정 위치로 이동
        
        Args:
            pos: 목표 위치 (0~100)
            reconnect_cb: 재연결 콜백
        
        Returns:
            에러 코드 (0: 성공)
        """
        robot = self.connector.get_robot()
        if robot is None:
            return -1
        
        try:
            err = self.robot_state.safe_call(
                robot.MoveGripper,
                cfg.GRIPPER_INDEX,
                int(pos),
                int(cfg.GRIPPER_SPEED),
                int(cfg.GRIPPER_FORCE),
                int(cfg.GRIPPER_MAX_TIME),
                int(cfg.GRIPPER_BLOCK),
                0, 0, 0, 0,
                reconnect_cb=reconnect_cb
            )
            return int(err)
        except Exception as e:
            print(f"[GRIP-FAIL] MoveGripper exception: {e}")
            return -999
    
    # -------------------------
    # Open/Close/Toggle
    # -------------------------
    def open(self, reconnect_cb: Optional[Callable] = None) -> bool:
        """
        그리퍼 열기
        
        Args:
            reconnect_cb: 재연결 콜백
        
        Returns:
            bool: 성공 시 True
        """
        if not self.ensure_activated(reconnect_cb=reconnect_cb):
            return False
        
        print("[GRIP] Opening gripper...")
        err = self.move(cfg.GRIP_OPEN_POS, reconnect_cb=reconnect_cb)
        print(f"[GRIP] Open retval: {err}")
        
        if err == 0:
            self._state["gripper_closed"] = False
        
        time.sleep(0.3)
        return err == 0
    
    def close(self, pos: Optional[int] = None, reconnect_cb: Optional[Callable] = None) -> bool:
        """
        그리퍼 닫기
        
        Args:
            pos: 닫는 위치 (None이면 기본값 사용)
            reconnect_cb: 재연결 콜백
        
        Returns:
            bool: 성공 시 True
        """
        if not self.ensure_activated(reconnect_cb=reconnect_cb):
            return False
        
        close_pos = pos if pos is not None else cfg.GRIP_CLOSE_POS
        
        print(f"[GRIP] Closing gripper to {close_pos}...")
        err = self.move(close_pos, reconnect_cb=reconnect_cb)
        print(f"[GRIP] Close retval: {err}")
        
        if err == 0:
            self._state["gripper_closed"] = True
        
        time.sleep(0.3)
        return err == 0
    
    def toggle(self, reconnect_cb: Optional[Callable] = None) -> bool:
        """
        그리퍼 토글 (Open <-> Close)
        
        Args:
            reconnect_cb: 재연결 콜백
        
        Returns:
            bool: 성공 시 True
        """
        closed = self._state.get("gripper_closed", None)
        
        if closed is None:
            print("[GRIP] State unknown -> CLOSE first")
            return self.close(reconnect_cb=reconnect_cb)
        
        if closed:
            return self.open(reconnect_cb=reconnect_cb)
        else:
            return self.close(reconnect_cb=reconnect_cb)
    
    # -------------------------
    # 명령 (9번 기능 - 그리퍼 메뉴)
    # -------------------------
    def cmd_gripper_menu(self, reconnect_cb: Optional[Callable] = None):
        """
        그리퍼 제어 메뉴 (9번 메뉴)
        
        Args:
            reconnect_cb: 재연결 콜백
        """
        if not self.connector.is_connected():
            print("[9] 로봇이 연결되어 있지 않습니다. (0번으로 연결)")
            return
        
        while True:
            print("\n---------------------------------------")
            print("Gripper Control")
            print("  o : Open")
            print("  c : Close")
            print("  t : Toggle")
            print("  b : Back")
            print("---------------------------------------")
            
            cmd = input("gripper (o/c/t/b) > ").strip().lower()
            
            if cmd == "b":
                break
            elif cmd == "o":
                self.open(reconnect_cb=reconnect_cb)
            elif cmd == "c":
                self.close(reconnect_cb=reconnect_cb)
            elif cmd == "t":
                self.toggle(reconnect_cb=reconnect_cb)
            else:
                print("[WARN] Invalid gripper command")