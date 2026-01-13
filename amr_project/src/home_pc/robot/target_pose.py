# robot/target_pose.py
"""
목표 Pose 계산 클래스
- Vision 측정값 + 현재 로봇 pose → 목표 pose 계산
- 좌표 변환 및 보정 로직
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np

from config import robot_config as cfg


class TargetPose:
    """
    목표 Pose 계산 클래스
    - Vision 데이터와 현재 로봇 pose를 이용해 목표 pose 계산
    - Pivot 보정 및 각도 보정 적용
    """
    
    def __init__(self, robot_state, box_detector):
        """
        Args:
            robot_state: RobotState 인스턴스
            box_detector: BoxDetector 인스턴스
        """
        self.robot_state = robot_state
        self.box_detector = box_detector
        
        # 마지막 계산된 목표 pose 캐시
        self._last_target_pose6: Optional[List[float]] = None
        self._last_target_debug: Optional[Dict] = None
    
    # -------------------------
    # 유틸리티
    # -------------------------
    @staticmethod
    def ensure_pose6(pose) -> List[float]:
        """Pose6 검증 및 변환"""
        if not isinstance(pose, (list, tuple)) or len(pose) < 6:
            raise ValueError("pose6 must be list/tuple len>=6")
        return [float(x) for x in pose[:6]]
    
    @staticmethod
    def fmt_pose6(pose6) -> str:
        """Pose6 포맷팅"""
        x, y, z, rx, ry, rz = TargetPose.ensure_pose6(pose6)
        return f"[x,y,z,rx,ry,rz]=[{x:.3f}, {y:.3f}, {z:.3f}, {rx:.3f}, {ry:.3f}, {rz:.3f}]"
    
    # -------------------------
    # 핵심 계산 로직
    # -------------------------
    def compute_move_xyz_from_measure(self, measure_res: Dict, current_ry: float) -> Tuple[float, float, float]:
        """
        Vision 측정값으로부터 이동량(XYZ) 계산
        
        Args:
            measure_res: Vision 측정 결과
            current_ry: 현재 RY 각도
        
        Returns:
            (dx, dy, dz) 이동량 (mm)
        """
        # 카메라-그리퍼 오프셋 적용
        cx = float(measure_res["move_x_mm"]) + float(cfg.OFF_X_MM)
        cy = float(measure_res["move_y_mm"]) + float(cfg.OFF_Y_MM)
        cz = float(measure_res["move_z_mm"]) + float(cfg.OFF_Z_MM)
        
        # 초기 변환
        dx0 = -cx
        dy0 = cy
        dz0 = -cz
        
        # RY 회전 보정
        rad_ry = np.deg2rad(float(current_ry))
        c, s = float(np.cos(rad_ry)), float(np.sin(rad_ry))
        
        dx1 = dx0
        dy1 = dy0 * c + dz0 * s
        dz1 = -dy0 * s + dz0 * c
        
        # 베이스 Yaw 오프셋 적용
        yaw_deg = float(cfg.BASE_YAW_OFFSET_DEG)
        rad_yaw = np.deg2rad(yaw_deg)
        c_y, s_y = float(np.cos(rad_yaw)), float(np.sin(rad_yaw))
        
        dx_final = c_y * dx1 - s_y * dy1
        dy_final = s_y * dx1 + c_y * dy1
        dz_final = dz1
        
        return float(dx_final), float(dy_final), float(dz_final)
    
    def build_target_pose(self, current_tcp_pose6, measure_res: Dict) -> List[float]:
        """
        목표 pose 계산
        
        Args:
            current_tcp_pose6: 현재 TCP pose
            measure_res: Vision 측정 결과
        
        Returns:
            목표 pose6
        """
        x, y, z, rx, ry, rz = self.ensure_pose6(current_tcp_pose6)
        
        # 이동량 계산
        dx, dy, dz = self.compute_move_xyz_from_measure(measure_res, ry)
        target_x = x + dx
        target_y = y + dy
        target_z = z + dz
        
        # Pivot 길이 및 목표 RY
        L = float(cfg.PIVOT_LENGTH)
        target_ry = float(cfg.TARGET_RY_DEG)
        
        # Pivot 보정 계산
        rad_curr = np.deg2rad(float(ry))
        rad_targ = np.deg2rad(float(target_ry))
        
        comp_y_local = L * (float(np.sin(rad_curr)) - float(np.sin(rad_targ)))
        comp_z_local = L * (float(np.cos(rad_targ)) - float(np.cos(rad_curr)))
        
        # Yaw 회전 적용
        yaw_deg = float(cfg.BASE_YAW_OFFSET_DEG)
        rad_yaw = np.deg2rad(yaw_deg)
        
        comp_dx = -float(np.sin(rad_yaw)) * comp_y_local
        comp_dy = float(np.cos(rad_yaw)) * comp_y_local
        
        # 박스 각도 보정
        box_angle = float(measure_res.get("angle_deg", 0.0))
        target_rz = float(rz) - box_angle
        
        # 최종 목표 pose
        final_x = target_x - comp_dx
        final_y = target_y - comp_dy
        final_z = target_z - comp_z_local
        
        return [float(final_x), float(final_y), float(final_z), 
                float(rx), float(target_ry), float(target_rz)]
    
    def build_target_pose_with_debug(self, current_tcp_pose6, measure_res: Dict) -> Dict:
        """
        목표 pose 계산 (디버그 정보 포함)
        
        Args:
            current_tcp_pose6: 현재 TCP pose
            measure_res: Vision 측정 결과
        
        Returns:
            디버그 정보가 포함된 딕셔너리
        """
        x, y, z, rx, ry, rz = self.ensure_pose6(current_tcp_pose6)
        target = self.build_target_pose(current_tcp_pose6, measure_res)
        
        return {
            "target_pose6": target,
            "move_x_mm": float(target[0] - x),
            "move_y_mm": float(target[1] - y),
            "move_z_mm": float(target[2] - z),
            "current_ry": float(ry),
            "target_ry": float(target[4]),
            "angle_deg": float(measure_res.get("angle_deg", 0.0)),
            "cur_pose6": [float(x), float(y), float(z), float(rx), float(ry), float(rz)],
            "measure": dict(measure_res),
        }
    
    # -------------------------
    # 캐시 관리
    # -------------------------
    def set_last_target_pose6(self, pose6: Optional[List[float]], 
                              debug: Optional[Dict] = None):
        """마지막 목표 pose 저장"""
        self._last_target_pose6 = None if pose6 is None else self.ensure_pose6(pose6)
        self._last_target_debug = debug
    
    def get_last_target_pose6(self) -> Optional[List[float]]:
        """마지막 목표 pose 반환"""
        return self._last_target_pose6
    
    def get_last_target_debug(self) -> Optional[Dict]:
        """마지막 디버그 정보 반환"""
        return self._last_target_debug
    
    # -------------------------
    # 명령 (4번 기능)
    # -------------------------
    def cmd_build_target_from_last(self, use_last_pose: bool = True, 
                                    reconnect_cb=None) -> Optional[Dict]:
        """
        마지막 측정값으로 목표 pose 계산 (4번 메뉴)
        
        Args:
            use_last_pose: True면 캐시된 pose 사용, False면 새로 읽기
            reconnect_cb: 재연결 콜백
        
        Returns:
            계산 결과 딕셔너리 또는 None
        """
        # Vision 측정값 가져오기
        meas = self.box_detector.get_last_measure_avg()
        if meas is None:
            print("[Target] Vision 측정값이 없습니다. (3번으로 먼저 측정)")
            self.set_last_target_pose6(None, None)
            return None
        
        # 현재 pose 가져오기
        pose = None
        if use_last_pose:
            # 캐시된 pose 사용
            pose, _ = self.robot_state.get_last_pose_joint()
            if pose is None:
                print("[Target] 로봇 pose가 없습니다. (2번으로 먼저 읽기)")
                self.set_last_target_pose6(None, None)
                return None
        else:
            # 새로 읽기
            (err_p, pose), _ = self.robot_state.read_pose_joint(reconnect_cb=reconnect_cb)
            if err_p != 0 or pose is None:
                print("[Target] 로봇 pose 읽기 실패")
                self.set_last_target_pose6(None, None)
                return None
        
        # 목표 pose 계산
        try:
            debug = self.build_target_pose_with_debug(pose, meas)
            self.set_last_target_pose6(debug["target_pose6"], debug)
            
            print("[Target] 목표 Pose 계산 완료:")
            print(f"  Current: {self.fmt_pose6(debug['cur_pose6'])}")
            print(f"  Target:  {self.fmt_pose6(debug['target_pose6'])}")
            print(f"  Move(mm): dx={debug['move_x_mm']:.1f}, "
                  f"dy={debug['move_y_mm']:.1f}, dz={debug['move_z_mm']:.1f}")
            print(f"  Angle: {debug['angle_deg']:.1f}° → RZ={debug['target_pose6'][5]:.1f}°")
            
            return debug
        
        except Exception as e:
            print(f"[Target] 계산 실패: {e}")
            self.set_last_target_pose6(None, None)
            return None