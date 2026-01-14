# robot/robot_motion.py
"""
로봇 모션 제어 클래스
- MoveCart (직교 좌표 이동)
- MoveJ (관절 공간 이동)
- 속도 fallback 지원
"""
from __future__ import annotations
import time
from typing import List, Optional, Sequence, Callable

from config import robot_config as cfg


class RobotMotion:
    """
    로봇 모션 제어 클래스
    - MoveCart: 직교 좌표계 이동
    - MoveJ: 관절 공간 이동
    - 안전한 재시도 및 속도 fallback
    """
    
    def __init__(self, robot_connector, robot_state):
        """
        Args:
            robot_connector: RobotConnector 인스턴스
            robot_state: RobotState 인스턴스
        """
        self.connector = robot_connector
        self.robot_state = robot_state
    
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
    def ensure_joint6(j) -> List[float]:
        """Joint6 검증"""
        if not isinstance(j, (list, tuple)) or len(j) < 6:
            raise ValueError("joint6 must be list/tuple len>=6")
        return [float(x) for x in j[:6]]
    
    @staticmethod
    def clamp(v: float, lo: float, hi: float) -> float:
        """값 제한"""
        return max(float(lo), min(float(hi), float(v)))
    
    # -------------------------
    # MoveCart (직교 좌표 이동)
    # -------------------------
    def move_cart(self, pose6, 
                  tool: Optional[int] = None,
                  user: Optional[int] = None,
                  vel_list: Optional[Sequence[float]] = None,
                  acc: Optional[float] = None,
                  ovl: Optional[float] = None,
                  blendT: Optional[float] = None,
                  config: Optional[int] = None,
                  reconnect_cb: Optional[Callable] = None,
                  label: str = "") -> int:
        """
        MoveCart 실행 (속도 fallback 포함)
        
        Args:
            pose6: 목표 pose
            tool: Tool ID
            user: User ID
            vel_list: 속도 리스트 (fallback 순서대로)
            acc: 가속도
            ovl: Override
            blendT: Blending time
            config: Config
            reconnect_cb: 재연결 콜백
            label: 로그용 라벨
        
        Returns:
            에러 코드 (0: 성공)
        """
        robot = self.connector.get_robot()
        if robot is None:
            return -1
        
        pose6 = self.ensure_pose6(pose6)
        
        # 기본값 설정
        tool = int(cfg.TOOL_ID if tool is None else tool)
        user = int(cfg.USER_ID if user is None else user)
        vel_list = list(cfg.MOVE_CART_VEL_LIST if vel_list is None else vel_list)
        acc = float(cfg.MOVE_CART_ACC if acc is None else acc)
        ovl = float(cfg.MOVE_CART_OVL if ovl is None else ovl)
        blendT = float(cfg.MOVE_CART_BLENDT if blendT is None else blendT)
        config = int(cfg.MOVE_CART_EX if config is None else config)
        
        # 속도 fallback
        for vel in vel_list:
            rtn = self.robot_state.safe_call(
                robot.MoveCart,
                pose6, tool, user,
                float(vel), acc, ovl, blendT, config,
                reconnect_cb=reconnect_cb
            )
            
            if int(rtn) == 0:
                if label:
                    print(f"[MoveCart-OK] {label} vel={vel}")
                return 0
            
            # 112: 속도/제약 조건 실패 -> 다음 속도 시도
            if int(rtn) == 112:
                if label:
                    print(f"[MoveCart-112] {label} vel={vel} -> try next")
                continue
            
            if label:
                print(f"[MoveCart-FAIL] {label} vel={vel} err={rtn} -> try next")
        
        return 112
    
    # -------------------------
    # MoveJ (관절 공간 이동)
    # -------------------------
    def move_j(self, joint_pos,
               tool: Optional[int] = None,
               user: Optional[int] = None,
               vel: Optional[float] = None,
               blendT: Optional[float] = None,
               reconnect_cb: Optional[Callable] = None,
               label: str = "") -> int:
        """
        MoveJ 실행 (관절 공간 이동)
        
        Args:
            joint_pos: 목표 joint
            tool: Tool ID
            user: User ID
            vel: 속도
            blendT: Blending time
            reconnect_cb: 재연결 콜백
            label: 로그용 라벨
        
        Returns:
            에러 코드 (0: 성공)
        """
        robot = self.connector.get_robot()
        if robot is None:
            return -1
        
        joint_pos = self.ensure_joint6(joint_pos)
        
        # 기본값 설정
        tool = int(cfg.TOOL_ID if tool is None else tool)
        user = int(cfg.USER_ID if user is None else user)
        vel = float(cfg.MOVEJ_VEL_RETURN if vel is None else vel)
        blendT = float(cfg.MOVEJ_BLENDT_RETURN if blendT is None else blendT)
        
        rtn = self.robot_state.safe_call(
            robot.MoveJ,
            joint_pos=joint_pos,
            tool=tool,
            user=user,
            vel=vel,
            blendT=blendT,
            reconnect_cb=reconnect_cb
        )
        
        if label:
            status = "OK" if int(rtn) == 0 else f"FAIL(err={rtn})"
            print(f"[MoveJ-{status}] {label} vel={vel}")
        
        return int(rtn)
    
    # -------------------------
    # Joint 회전 (J4, J6 등)
    # -------------------------
    def rotate_joint(self, joint_index: int, delta_deg: float,
                     max_step_deg: Optional[float] = None,
                     vel: Optional[float] = None,
                     reconnect_cb: Optional[Callable] = None) -> int:
        """
        특정 Joint를 delta만큼 회전
        
        Args:
            joint_index: Joint 인덱스 (0~5, 0=J1, 5=J6)
            delta_deg: 회전 각도 (degree)
            max_step_deg: 최대 회전 각도 제한
            vel: 속도
            reconnect_cb: 재연결 콜백
        
        Returns:
            에러 코드 (0: 성공)
        """
        robot = self.connector.get_robot()
        if robot is None:
            return -1
        
        # 인덱스 검증
        if not (0 <= joint_index <= 5):
            print(f"[ERROR] Invalid joint_index: {joint_index} (must be 0~5)")
            return -2
        
        # Delta clamp
        if max_step_deg is not None:
            delta_deg = self.clamp(delta_deg, -max_step_deg, max_step_deg)
        
        # 현재 joint 읽기
        err_j, cur_joint = self.robot_state.read_joint6(reconnect_cb=reconnect_cb)
        if err_j != 0 or cur_joint is None:
            print(f"[ERROR] Failed to read current joint: err={err_j}")
            return err_j
        
        # 목표 joint 계산
        target_joint = list(cur_joint)
        target_joint[joint_index] = float(target_joint[joint_index]) + float(delta_deg)
        
        print(f"[ROTATE] J{joint_index+1}: "
              f"{cur_joint[joint_index]:.3f} -> {target_joint[joint_index]:.3f} "
              f"(delta {delta_deg:+.3f}°)")
        
        # MoveJ 실행
        return self.move_j(
            target_joint,
            vel=vel,
            reconnect_cb=reconnect_cb,
            label=f"J{joint_index+1}"
        )
    
    def rotate_j4(self, delta_deg: float, reconnect_cb: Optional[Callable] = None) -> int:
        """J4 회전"""
        return self.rotate_joint(
            joint_index=3,
            delta_deg=delta_deg,
            max_step_deg=getattr(cfg, "J4_MAX_STEP_DEG", 9999.0),
            vel=cfg.MOVEJ_VEL_J6,
            reconnect_cb=reconnect_cb
        )
    
    def rotate_j6(self, delta_deg: float, reconnect_cb: Optional[Callable] = None) -> int:
        """J6 회전"""
        return self.rotate_joint(
            joint_index=5,
            delta_deg=delta_deg,
            max_step_deg=cfg.J6_MAX_STEP_DEG,
            vel=cfg.MOVEJ_VEL_J6,
            reconnect_cb=reconnect_cb
        )