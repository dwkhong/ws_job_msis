# robot/robot_connector.py
"""
Fairino 로봇 연결 관리 클래스
로봇 연결, 연결 해제, 상태 확인을 담당
"""
import sys
from typing import Optional


class RobotConnector:
    """
    Fairino 로봇 연결 관리 클래스
    - 로봇 연결/해제만 담당
    - 다른 기능(모션, 그리퍼 등)은 별도 클래스로 분리
    
    Attributes:
        ip (str): 로봇 IP 주소
        robot: Robot.RPC 인스턴스 (연결 시)
        sdk_path (str): Fairino SDK 경로
    """
    
    def __init__(self, ip: str, sdk_path: str):
        """
        Args:
            ip: 로봇 IP 주소
            sdk_path: Fairino Python SDK 경로
        """
        self.ip = ip
        self.sdk_path = sdk_path
        self.robot: Optional[object] = None
        
        # SDK 경로를 sys.path에 추가
        if self.sdk_path not in sys.path:
            sys.path.insert(0, self.sdk_path)
    
    def is_connected(self) -> bool:
        """
        로봇 연결 상태 확인
        
        Returns:
            bool: 연결되어 있으면 True
        """
        return self.robot is not None
    
    def connect(self) -> bool:
        """
        로봇에 연결
        
        Returns:
            bool: 연결 성공 시 True
        """
        if self.is_connected():
            print(f"[ROBOT] Already connected ✅ ip={self.ip}")
            return True
        
        try:
            # Robot 모듈 임포트 (SDK 경로가 sys.path에 추가된 후)
            import Robot
            
            print(f"[ROBOT] Connecting... ip={self.ip}")
            self.robot = Robot.RPC(self.ip)
            print("[ROBOT] Connected ✅")
            
            # 속도 설정 (중요!)
            try:
                ret = self.robot.SetSpeed(80)
                if ret == 0:
                    print("[ROBOT] Speed set to 80 ✅")
                else:
                    print(f"[ROBOT] SetSpeed(80) failed: ret={ret}")
            except Exception as e:
                print(f"[ROBOT] SetSpeed error: {e}")
            
            return True
            
        except Exception as e:
            print(f"[ROBOT] Connection failed: {e}")
            self.robot = None
            return False
    
    def disconnect(self) -> bool:
        """
        로봇 연결 해제
        
        Returns:
            bool: 연결 해제 성공 시 True
        """
        if not self.is_connected():
            print("[ROBOT] Not connected.")
            return False
        
        print("[ROBOT] Closing...")
        try:
            self.robot.CloseRPC()
            print("[ROBOT] Closed ✅")
        except Exception as e:
            print(f"[ROBOT] Error during disconnect: {e}")
        finally:
            self.robot = None
        
        return True
    
    def toggle(self) -> bool:
        """
        연결 상태 토글 (연결 <-> 연결 해제)
        
        Returns:
            bool: 토글 후 연결 상태 (True: 연결됨, False: 연결 해제됨)
        """
        if self.is_connected():
            self.disconnect()
            return False
        else:
            return self.connect()
    
    def get_robot(self):
        """
        Robot.RPC 인스턴스 반환
        
        Returns:
            Robot.RPC 인스턴스 또는 None
        """
        return self.robot
    
    # -------------------------
    # 로봇 제어 (GUI용)
    # -------------------------
    def stop_motion(self):
        """로봇 모션 즉시 정지"""
        if self.robot:
            try:
                self.robot.StopMotion()
                print("[ROBOT] Motion stopped")
            except Exception as e:
                print(f"[ROBOT] Stop failed: {e}")
    
    def pause_motion(self):
        """로봇 모션 일시 정지"""
        if self.robot:
            try:
                self.robot.PauseMotion()
                print("[ROBOT] Motion paused")
            except Exception as e:
                print(f"[ROBOT] Pause failed: {e}")
    
    def resume_motion(self):
        """로봇 모션 재개"""
        if self.robot:
            try:
                self.robot.ResumeMotion()
                print("[ROBOT] Motion resumed")
            except Exception as e:
                print(f"[ROBOT] Resume failed: {e}")