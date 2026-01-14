# robot/ik_checker.py
"""
IK 검증 클래스
- 목표 pose의 IK 해 존재 확인
- Phase0 (중간 위치) IK 검증
- RX/RY 각도 탐색으로 해 찾기
"""
from __future__ import annotations
import time
from typing import Any, Dict, List, Optional, Tuple, Callable

from config import robot_config as cfg


class IKChecker:
    """
    IK 검증 클래스
    - 목표 pose가 도달 가능한지 확인
    - Phase0 (Z를 올린 중간 위치) 검증
    - 각도 탐색을 통한 대안 pose 찾기
    """
    
    def __init__(self, robot_connector, robot_state, target_pose):
        """
        Args:
            robot_connector: RobotConnector 인스턴스
            robot_state: RobotState 인스턴스
            target_pose: TargetPose 인스턴스
        """
        self.connector = robot_connector
        self.robot_state = robot_state
        self.target_pose = target_pose
        
        # 마지막 IK 검증 결과 캐시
        self._last_ik_result: Optional[Dict[str, Any]] = None
    
    # -------------------------
    # 유틸리티
    # -------------------------
    @staticmethod
    def ensure_pose6(p) -> List[float]:
        """Pose6 검증 및 변환"""
        if not isinstance(p, (list, tuple)) or len(p) < 6:
            raise ValueError("pose6 must be list/tuple len>=6")
        return [float(x) for x in p[:6]]
    
    @staticmethod
    def ensure_joint6(j) -> List[float]:
        """Joint6 검증 및 변환"""
        if not isinstance(j, (list, tuple)) or len(j) < 6:
            raise ValueError("joint6 must be list/tuple len>=6")
        return [float(x) for x in j[:6]]
    
    @staticmethod
    def is_timeout_like(e: Exception) -> bool:
        """타임아웃 에러 판별"""
        s = str(e).lower()
        return ("timeout" in s) or ("timed out" in s) or ("实时数据失败" in str(e))
    
    @staticmethod
    def fmt_pose6(pose6) -> str:
        """Pose6 포맷팅"""
        x, y, z, rx, ry, rz = IKChecker.ensure_pose6(pose6)
        return f"[x,y,z,rx,ry,rz]=[{x:.3f}, {y:.3f}, {z:.3f}, {rx:.3f}, {ry:.3f}, {rz:.3f}]"
    
    @staticmethod
    def wrap_deg_180(a: float) -> float:
        """각도를 -180~180 범위로 정규화"""
        a = float(a)
        return (a + 180.0) % 360.0 - 180.0
    
    @staticmethod
    def target_variants_rz_ry(target_pose6, ry_cands) -> List[List[float]]:
        """
        Target pose의 RZ/RY 변형 생성
        - RZ: wrap(-180~180) 후 ±360 표현 후보
        - RY: 주어진 후보 리스트
        
        Args:
            target_pose6: 목표 pose
            ry_cands: RY 후보 리스트
        
        Returns:
            변형된 pose 리스트
        """
        t = IKChecker.ensure_pose6(target_pose6)
        base_rx = float(t[3])
        base_ry = float(t[4])
        base_rz = float(t[5])
        
        rz_wrap = IKChecker.wrap_deg_180(base_rz)
        rz_list = [rz_wrap, rz_wrap + 360.0, rz_wrap - 360.0]
        
        out: List[List[float]] = []
        seen = set()
        
        # 원래 ry가 후보 리스트에 없으면 맨 앞에 추가
        ry_list = list(ry_cands) if ry_cands else [base_ry]
        if all(abs(float(x) - base_ry) > 1e-6 for x in ry_list):
            ry_list = [base_ry] + ry_list
        
        for ry in ry_list:
            for rz in rz_list:
                cand = t.copy()
                cand[3] = base_rx
                cand[4] = float(ry)
                cand[5] = float(rz)
                
                # 중복 제거 키(ry, wrap(rz))
                key = (round(cand[4], 3), round(IKChecker.wrap_deg_180(cand[5]), 3))
                if key in seen:
                    continue
                seen.add(key)
                out.append(cand)
        
        return out
    
    # -------------------------
    # Safe RPC 호출
    # -------------------------
    def safe_call(self, fn: Callable, *args, reconnect_cb: Optional[Callable] = None, **kwargs):
        """
        RPC 호출 with 재시도
        
        Args:
            fn: 호출할 함수
            reconnect_cb: 재연결 콜백
        
        Returns:
            함수 실행 결과
        """
        last_e = None
        retry = int(cfg.RPC_RETRY)
        sleep_sec = float(cfg.RPC_RETRY_SLEEP_SEC)
        
        for k in range(retry + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_e = e
                # 타임아웃이면 재연결 시도
                if self.is_timeout_like(e) and reconnect_cb is not None:
                    try:
                        reconnect_cb()
                    except Exception:
                        pass
                if k < retry:
                    time.sleep(sleep_sec)
                    continue
                raise last_e
    
    # -------------------------
    # IK 검증
    # -------------------------
    def has_solution(self, pose6, cur_joint6, reconnect_cb: Optional[Callable] = None) -> bool:
        """
        IK 해가 존재하는지 확인
        
        Args:
            pose6: 목표 pose
            cur_joint6: 현재 joint
            reconnect_cb: 재연결 콜백
        
        Returns:
            해가 존재하면 True
        """
        robot = self.connector.get_robot()
        if robot is None:
            return False
        
        pose6 = self.ensure_pose6(pose6)
        cur_joint6 = self.ensure_joint6(cur_joint6)
        
        err, ok = self.safe_call(
            robot.GetInverseKinHasSolution,
            int(cfg.IK_REF),
            pose6,
            cur_joint6,
            reconnect_cb=reconnect_cb,
        )
        return (int(err) == 0) and bool(ok)
    
    # -------------------------
    # IK 검증 (Phase0 포함)
    # -------------------------
    def check_target_ik(self, cur_joint6, target_pose6,
                        check_phase0: bool = True,
                        z_hold_offset_mm: Optional[float] = None,
                        reconnect_cb: Optional[Callable] = None) -> Dict[str, Any]:
        """
        목표 pose의 IK 검증 (Phase0 포함)
        - RZ wrap/±360 재시도
        - RY 후보 재시도
        - 성공한 ok_target_pose6를 결과에 포함
        
        Args:
            cur_joint6: 현재 joint
            target_pose6: 목표 pose
            check_phase0: Phase0 검증 여부
            z_hold_offset_mm: Phase0 Z 오프셋
            reconnect_cb: 재연결 콜백
        
        Returns:
            검증 결과 딕셔너리
        """
        robot = self.connector.get_robot()
        if robot is None:
            return {"ok": False, "msg": "robot is None"}
        
        cur_joint6 = self.ensure_joint6(cur_joint6)
        target_in = self.ensure_pose6(target_pose6)
        
        # RY 후보 리스트
        ry_cands = list(getattr(cfg, "TARGET_RY_CAND_LIST", [2.0, 1.0, 0.0]))
        
        # 0) Target IK (RZ wrap/±360 + RY 후보로 재시도)
        ok_target: Optional[List[float]] = None
        tries = 0
        
        try:
            for cand in self.target_variants_rz_ry(target_in, ry_cands):
                tries += 1
                if self.has_solution(cand, cur_joint6, reconnect_cb=reconnect_cb):
                    ok_target = cand
                    break
        except Exception as e:
            return {
                "ok": False,
                "msg": f"target IK check exception: {e}",
                "target_pose6": target_in,
                "ok_target_pose6": None,
                "phase0_pose6": None,
            }
        
        if ok_target is None:
            return {
                "ok": False,
                "target_ok": False,
                "phase0_ok": False,
                "target_pose6": target_in,
                "ok_target_pose6": None,
                "phase0_pose6": None,
                "mode": "TARGET_FAIL",
                "tries": tries,
                "msg": "target IK not solvable (tried rz wrap/±360 + ry candidates)",
            }
        
        if ok_target is None:
            return {
                "ok": False,
                "target_ok": False,
                "phase0_ok": False,
                "target_pose6": target_in,
                "ok_target_pose6": None,
                "phase0_pose6": None,
                "mode": "TARGET_FAIL",
                "tries": tries,
                "msg": "target IK not solvable (tried rz wrap/±360 + ry candidates)",
            }
        
        # ok_target 기준으로 phase0 진행
        if not check_phase0:
            return {
                "ok": True,
                "target_ok": True,
                "phase0_ok": True,
                "target_pose6": target_in,
                "ok_target_pose6": ok_target,
                "phase0_pose6": None,
                "mode": "NO_PHASE0",
                "tries": tries,
                "msg": "target OK (phase0 skipped)",
            }
        
        zoff = float(cfg.Z_HOLD_OFFSET_MM if z_hold_offset_mm is None else z_hold_offset_mm)
        
        # 1) Phase0 strict: ok_target + Z만 위로
        phase0_strict = list(ok_target)
        phase0_strict[2] = float(phase0_strict[2] + zoff)
        
        try:
            tries += 1
            if self.has_solution(phase0_strict, cur_joint6, reconnect_cb=reconnect_cb):
                return {
                    "ok": True,
                    "target_ok": True,
                    "phase0_ok": True,
                    "target_pose6": target_in,
                    "ok_target_pose6": ok_target,
                    "phase0_pose6": phase0_strict,
                    "mode": "STRICT",
                    "tries": tries,
                    "msg": "target OK, phase0 OK (strict)",
                }
        except Exception as e:
            return {
                "ok": False,
                "msg": f"phase0 strict IK exception: {e}",
                "target_pose6": target_in,
                "ok_target_pose6": ok_target,
                "phase0_pose6": None,
            }
        
        # 2) Strict가 안 되면: Phase0에서만 RX/RY 스윕 (RZ는 유지)
        rx_list = list(cfg.SEARCH_RX_LIST)
        ry_list = list(cfg.SEARCH_RY_LIST)
        
        base_rx = float(ok_target[3])
        base_ry = float(ok_target[4])
        base_rz = float(ok_target[5])
        
        timeout_sec = float(cfg.SEARCH_TIMEOUT_SEC)
        max_tries = int(cfg.SEARCH_MAX_TRIES)
        t0 = time.time()
        
        def timed_out() -> bool:
            return (time.time() - t0) > timeout_sec
        
        best: Optional[List[float]] = None
        
        for dry in ry_list:
            for drx in rx_list:
                tries += 1
                if tries > max_tries or timed_out():
                    break
                
                cand = list(phase0_strict)
                cand[3] = base_rx + float(drx)
                cand[4] = base_ry + float(dry)
                cand[5] = base_rz  # RZ 유지
                
                try:
                    if self.has_solution(cand, cur_joint6, reconnect_cb=reconnect_cb):
                        best = cand
                        break
                except Exception:
                    continue
            
            if best is not None or tries > max_tries or timed_out():
                break
        
        if best is None:
            return {
                "ok": False,
                "target_ok": True,
                "phase0_ok": False,
                "target_pose6": target_in,
                "ok_target_pose6": ok_target,
                "phase0_pose6": None,
                "mode": "PHASE0_FAIL",
                "tries": tries,
                "msg": "target OK but phase0 IK not solvable",
            }
        
        return {
            "ok": True,
            "target_ok": True,
            "phase0_ok": True,
            "target_pose6": target_in,
            "ok_target_pose6": ok_target,
            "phase0_pose6": best,
            "mode": "SEARCH_TILT",
            "tries": tries,
            "msg": "target OK, phase0 OK (rx/ry sweep)",
        }
    
    # -------------------------
    # 캐시 관리
    # -------------------------
    def get_last_ik_result(self) -> Optional[Dict[str, Any]]:
        """마지막 IK 결과 반환"""
        return self._last_ik_result
    
    def get_last_phase0_pose6(self) -> Optional[List[float]]:
        """마지막 Phase0 pose 반환"""
        if self._last_ik_result and self._last_ik_result.get("ok"):
            return self._last_ik_result.get("phase0_pose6")
        return None
    
    def get_last_ok_target_pose6(self) -> Optional[List[float]]:
        """
        실제로 IK 해가 있는 target pose 반환
        (RZ wrap/±360 또는 RY 후보로 성공한 pose)
        """
        if self._last_ik_result and self._last_ik_result.get("ok"):
            return self._last_ik_result.get("ok_target_pose6")
        return None
    
    # -------------------------
    # 명령 (5번 기능)
    # -------------------------
    def cmd_check_target_from_last(self, check_phase0: bool = True,
                                    reconnect_cb: Optional[Callable] = None) -> Dict[str, Any]:
        """
        마지막 목표 pose IK 검증 (5번 메뉴)
        
        Args:
            check_phase0: Phase0 검증 여부
            reconnect_cb: 재연결 콜백
        
        Returns:
            검증 결과 딕셔너리
        """
        if not self.connector.is_connected():
            print("[IK] 로봇이 연결되어 있지 않습니다. (0번으로 연결)")
            self._last_ik_result = {"ok": False, "msg": "robot is None"}
            return self._last_ik_result
        
        # Joint 캐시 가져오기
        last_joint = self.robot_state.get_last_joint6()
        if last_joint is None:
            print("[IK] last_joint6 is None (2번 먼저 실행)")
            self._last_ik_result = {"ok": False, "msg": "last_joint6 is None"}
            return self._last_ik_result
        
        # Target 캐시 가져오기
        last_target = self.target_pose.get_last_target_pose6()
        if last_target is None:
            print("[IK] last_target_pose6 is None (4번 먼저 실행)")
            self._last_ik_result = {"ok": False, "msg": "last_target_pose6 is None"}
            return self._last_ik_result
        
        # IK 검증
        res = self.check_target_ik(
            cur_joint6=last_joint,
            target_pose6=last_target,
            check_phase0=check_phase0,
            reconnect_cb=reconnect_cb,
        )
        
        # 결과 캐시 저장
        self._last_ik_result = res
        
        # 결과 출력
        if res.get("ok"):
            print(f"[IK] ✅ OK  mode={res.get('mode')}  tries={res.get('tries')}")
            print("[IK] target(in):", self.fmt_pose6(res["target_pose6"]))
            if res.get("ok_target_pose6") is not None:
                print("[IK] target(ok):", self.fmt_pose6(res["ok_target_pose6"]))
            if res.get("phase0_pose6") is not None:
                print("[IK] phase0    :", self.fmt_pose6(res["phase0_pose6"]))
        else:
            print(f"[IK] ❌ FAIL: {res.get('msg')}")
            if res.get("target_pose6") is not None:
                print("[IK] target :", self.fmt_pose6(res["target_pose6"]))
        
        return res