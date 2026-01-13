# robot/robot_state.py
"""
로봇 상태 관리 클래스
- 현재 pose/joint 읽기
- 초기 joint 저장
- 캐시 관리
"""
import time
from typing import Optional, Tuple, List, Dict, Callable
from config import robot_config as cfg


class RobotState:
    """
    로봇 상태 관리 클래스
    - TCP pose, joint 읽기
    - 마지막 상태 캐시
    - 초기 joint 저장
    """
    
    def __init__(self, robot_connector):
        """
        Args:
            robot_connector: RobotConnector 인스턴스
        """
        self.connector = robot_connector
        
        # 상태 캐시
        self._last_pose: Optional[List[float]] = None
        self._last_joint: Optional[List[float]] = None
        self._initial_joint: Optional[List[float]] = None
    
    # -------------------------
    # 유틸리티 함수
    # -------------------------
    @staticmethod
    def fmt_pose6(pose: List[float]) -> str:
        """Pose6 포맷팅"""
        if not isinstance(pose, (list, tuple)) or len(pose) < 6:
            return str(pose)
        x, y, z, rx, ry, rz = pose[:6]
        return f"[x,y,z,rx,ry,rz]=[{x:.3f}, {y:.3f}, {z:.3f}, {rx:.3f}, {ry:.3f}, {rz:.3f}]"
    
    @staticmethod
    def fmt_joint(joint: List[float]) -> str:
        """Joint6 포맷팅"""
        if not isinstance(joint, (list, tuple)) or len(joint) < 6:
            return str(joint)
        return "[" + ", ".join(f"{float(v):.3f}" for v in joint[:6]) + "]"
    
    @staticmethod
    def ensure_pose6(p) -> List[float]:
        """Pose6 검증 및 변환"""
        if not isinstance(p, (list, tuple)) or len(p) < 6:
            raise ValueError(f"pose invalid: {p}")
        return [float(x) for x in p[:6]]
    
    @staticmethod
    def ensure_joint6(j) -> List[float]:
        """Joint6 검증 및 변환"""
        if not isinstance(j, (list, tuple)) or len(j) < 6:
            raise ValueError(f"joint invalid: {j}")
        return [float(x) for x in j[:6]]
    
    # -------------------------
    # Safe RPC 호출
    # -------------------------
    def safe_call(self, fn: Callable, *args, 
                   retry: Optional[int] = None,
                   sleep_sec: Optional[float] = None,
                   reconnect_cb: Optional[Callable] = None,
                   **kwargs):
        """
        RPC 호출 with 재시도 및 타임아웃 처리
        
        Args:
            fn: 호출할 함수
            retry: 재시도 횟수
            sleep_sec: 재시도 간 대기 시간
            reconnect_cb: 재연결 콜백
        
        Returns:
            함수 실행 결과
        """
        retry = cfg.RPC_RETRY if retry is None else retry
        sleep_sec = cfg.RPC_RETRY_SLEEP_SEC if sleep_sec is None else sleep_sec
        last_e = None
        
        for k in range(retry + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_e = e
                msg = str(e).lower()
                
                timeout_like = ("timed out" in msg) or ("timeout" in msg) or ("实时数据失败" in str(e))
                
                # 재시도 가능하면 대기 후 재시도
                if k < retry:
                    if timeout_like:
                        print(f"[WARN] RPC timeout (retry {k+1}/{retry}): {e}")
                    time.sleep(sleep_sec)
                    continue
                
                # 마지막 시도 실패
                if timeout_like:
                    print(f"[WARN] RPC timeout (final): {e}")
                    if reconnect_cb is not None:
                        try:
                            reconnect_cb()
                        except Exception as e2:
                            print(f"[WARN] reconnect failed: {e2}")
                
                raise last_e
    
    # -------------------------
    # 상태 읽기
    # -------------------------
    def read_pose_joint(self, reconnect_cb: Optional[Callable] = None) -> Tuple[Tuple[int, Optional[List[float]]], Tuple[int, Optional[List[float]]]]:
        """
        현재 TCP pose와 joint 읽기
        
        Args:
            reconnect_cb: 재연결 콜백
        
        Returns:
            ((err_pose, pose6 or None), (err_joint, joint6 or None))
        """
        robot = self.connector.get_robot()
        if robot is None:
            return (1, None), (1, None)
        
        try:
            err_p, pose = self.safe_call(robot.GetActualTCPPose, flag=1, reconnect_cb=reconnect_cb)
            err_j, joint = self.safe_call(robot.GetActualJointPosDegree, flag=1, reconnect_cb=reconnect_cb)
            
            if err_p != 0 or err_j != 0:
                return (err_p, None), (err_j, None)
            
            return (0, self.ensure_pose6(pose)), (0, self.ensure_joint6(joint))
        
        except Exception as e:
            print(f"[ERROR] read_pose_joint failed: {e}")
            return (1, None), (1, None)
    
    def read_joint6(self, reconnect_cb: Optional[Callable] = None) -> Tuple[int, Optional[List[float]]]:
        """
        현재 joint만 읽기 (pose 없이)
        
        Args:
            reconnect_cb: 재연결 콜백
        
        Returns:
            (err, joint6 or None)
        """
        robot = self.connector.get_robot()
        if robot is None:
            return 1, None
        
        try:
            err_j, joint = self.safe_call(robot.GetActualJointPosDegree, flag=1, reconnect_cb=reconnect_cb)
            if int(err_j) != 0 or joint is None:
                return int(err_j), None
            return 0, self.ensure_joint6(joint)
        
        except Exception as e:
            print(f"[ERROR] read_joint6 failed: {e}")
            return 1, None
    
    # -------------------------
    # 초기 joint 관리
    # -------------------------
    def set_initial_joint6(self, joint6: List[float], force: bool = False):
        """
        초기 joint 저장
        
        Args:
            joint6: Joint 값
            force: 강제로 덮어쓰기
        """
        if joint6 is None:
            return
        if force or (self._initial_joint is None):
            self._initial_joint = self.ensure_joint6(joint6)
    
    def get_initial_joint6(self) -> Optional[List[float]]:
        """초기 joint 반환"""
        return self._initial_joint
    
    def try_capture_initial_joint(self, reconnect_cb: Optional[Callable] = None,
                                   force: bool = False, verbose: bool = True) -> Optional[List[float]]:
        """
        초기 joint 캡처 (한 번만 또는 force=True)
        
        Args:
            reconnect_cb: 재연결 콜백
            force: 강제로 다시 캡처
            verbose: 출력 여부
        
        Returns:
            초기 joint 또는 None
        """
        if (not force) and (self._initial_joint is not None):
            return self._initial_joint
        
        if not self.connector.is_connected():
            if verbose:
                print("[INIT] Robot not connected")
            return None
        
        err, j = self.read_joint6(reconnect_cb=reconnect_cb)
        if err == 0 and j is not None:
            self.set_initial_joint6(j, force=force)
            if verbose:
                print("[INIT] initial_joint6 saved:", self.fmt_joint(self._initial_joint))
            return self._initial_joint
        
        if verbose:
            print(f"[INIT] initial_joint6 capture failed err={err}")
        return None
    
    # -------------------------
    # 캐시 관리
    # -------------------------
    def get_last_pose_joint(self) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        """마지막 pose, joint 반환"""
        return self._last_pose, self._last_joint
    
    def get_last_joint6(self) -> Optional[List[float]]:
        """마지막 joint만 반환"""
        return self._last_joint
    
    # -------------------------
    # 명령 (2번 기능)
    # -------------------------
    def cmd_read_pose_joint(self, reconnect_cb: Optional[Callable] = None) -> Optional[Dict]:
        """
        현재 pose/joint 읽고 출력 (2번 메뉴)
        
        Args:
            reconnect_cb: 재연결 콜백
        
        Returns:
            결과 딕셔너리
        """
        if not self.connector.is_connected():
            print("[WARN] 로봇이 연결되어 있지 않습니다. (0번으로 연결)")
            return None
        
        (err_p, pose), (err_j, joint) = self.read_pose_joint(reconnect_cb=reconnect_cb)
        
        # Pose 출력
        if err_p == 0 and pose is not None:
            self._last_pose = pose
            print("tcp_pose :", self.fmt_pose6(pose))
        else:
            print(f"[ERR] GetActualTCPPose failed: err={err_p}")
        
        # Joint 출력
        if err_j == 0 and joint is not None:
            self._last_joint = joint
            
            # 초기값이 없으면 저장
            self.set_initial_joint6(joint, force=False)
            
            print("joint6   :", self.fmt_joint(joint))
        else:
            print(f"[ERR] GetActualJointPosDegree failed: err={err_j}")
        
        return {
            "err_pose": err_p,
            "pose": pose,
            "err_joint": err_j,
            "joint": joint,
        }