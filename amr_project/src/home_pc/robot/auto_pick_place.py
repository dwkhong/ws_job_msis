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
    def rotate_vertical_box_and_place(self, home_joint6, reconnect_cb: Optional[Callable] = None) -> Dict[str, Any]:
        """
        세로 박스 회전 (J5 기울이기 방식)
        1) 그리퍼 추가로 조이기 (25 → 18)
        2) J5 회전 (박스 기울이기)
        3) 그리퍼 열기
        4) 박스가 자동으로 눕게 됨
        5) HOME 복귀
        
        Args:
            home_joint6: Home joint 위치
            reconnect_cb: 재연결 콜백
        
        Returns:
            결과 딕셔너리
        """
        print("\n[ROTATE] 세로 박스 회전 시작 (J5 기울이기)")
        
        # 1) 그리퍼 추가로 조이기 (25 → cfg.GRIP_VERTICAL_CLOSE_POS)
        print(f"[ROTATE] Step 1: 그리퍼 추가 조이기 (25 → {cfg.GRIP_VERTICAL_CLOSE_POS})")
        try:
            err = self.gripper.move(
                pos=cfg.GRIP_VERTICAL_CLOSE_POS,
                reconnect_cb=reconnect_cb
            )
            if int(err) != 0:
                return {"ok": False, "msg": f"gripper move failed: err={err}"}
            
            time.sleep(cfg.GRIPPER_SETTLE_TIME_VERTICAL)  # 안정화
            print(f"[ROTATE] ✅ 그리퍼 {cfg.GRIP_VERTICAL_CLOSE_POS}로 조임 완료")
        except Exception as e:
            return {"ok": False, "msg": f"gripper tighten failed: {e}"}
        
        # 2) 현재 Joint 읽기
        (e, pose), (e_j, joint) = self.robot_state.read_pose_joint(reconnect_cb)
        if e_j != 0 or joint is None:
            return {"ok": False, "msg": "read joint failed"}
        
        # 3) J5 회전 (박스 기울이기)
        print("[ROTATE] Step 2: J5 회전 (박스 기울이기)")
        
        # J5 회전 (config에서 가져오기)
        j5_rotate_angle = cfg.J5_VERTICAL_ROTATE_ANGLE
        target_j5 = joint[4] + j5_rotate_angle
        
        # 각도 정규화 (-180 ~ 180)
        if target_j5 > 180:
            target_j5 -= 360
        elif target_j5 < -180:
            target_j5 += 360
        
        # J5 회전 실행
        new_joint = joint.copy()
        new_joint[4] = target_j5
        
        r = self.robot_motion.move_j(
            joint_pos=new_joint,
            vel=cfg.MOVEJ_VEL_J4,
            blendT=cfg.MOVEJ_BLENDT_J4,
            label="tilt_box",
            reconnect_cb=reconnect_cb
        )
        
        if int(r) != 0:
            print(f"[ROTATE] ❌ J5 회전 실패 err={r}")
            return {"ok": False, "msg": f"j5 rotate failed err={r}"}
        
        print(f"[ROTATE] ✅ J5 회전 완료 (박스 기울어짐, 각도={j5_rotate_angle}도)")
        time.sleep(cfg.J5_VERTICAL_SETTLE_TIME)  # 박스 안정화
        
        # 4) 그리퍼 열기 (박스가 자동으로 눕게 됨)
        print("[ROTATE] Step 3: 그리퍼 열기 (박스 자동 낙하)")
        try:
            self.gripper.open(reconnect_cb=reconnect_cb)
            time.sleep(cfg.GRIPPER_SETTLE_TIME_OPEN)
            print("[ROTATE] ✅ 그리퍼 열기 완료 (박스 눕혀짐)")
        except Exception as e:
            return {"ok": False, "msg": f"gripper open failed: {e}"}
        
        # 5) HOME 복귀
        print("[ROTATE] Step 4: HOME 복귀")
        r = self.robot_motion.move_j(
            joint_pos=home_joint6,
            label="HOME after tilt",
            reconnect_cb=reconnect_cb
        )
        if int(r) != 0:
            return {"ok": False, "msg": f"home failed err={r}"}
        
        print("[ROTATE] ✅ 세로 박스 회전 완료 (박스가 눕혀짐)")
        return {"ok": True, "msg": "rotate done (box laid down)"}
    
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
        
        # 스택 위치 선택 (1~3개: 위치1, 4~6개: 위치2)
        if self._stack_counter < cfg.STACK_POSITION_CHANGE_AT:
            # 위치 1 사용
            a_joint = cfg.WP11_A_JOINT
            drop_base = cfg.WP11_DROP_BASE_POSE
            stack_offset = self._stack_counter
            position_name = "위치1"
        else:
            # 위치 2 사용
            a_joint = cfg.WP11_B_A_JOINT
            drop_base = cfg.WP11_B_DROP_BASE_POSE
            stack_offset = self._stack_counter - cfg.STACK_POSITION_CHANGE_AT
            position_name = "위치2"
        
        # Drop 위치 계산 (스택 높이 반영)
        drop = list(drop_base)
        drop[2] = float(drop[2]) + float(cfg.STACK_Z_STEP_MM) * stack_offset
        
        # 높이 제한 체크
        if cfg.STACK_Z_MAX_MM is not None and float(drop[2]) > float(cfg.STACK_Z_MAX_MM):
            return {
                "ok": False,
                "msg": f"DROP Z too high: {drop[2]:.1f} > {float(cfg.STACK_Z_MAX_MM):.1f}",
                "drop": drop
            }
        
        print(f"\n[PLACE] counter={self._stack_counter} {position_name} stack_offset={stack_offset} dropZ={drop[2]:.1f}")
        
        # 1) A 위치로 이동
        r = self.robot_motion.move_j(
            joint_pos=a_joint,
            label="A",
            reconnect_cb=reconnect_cb
        )
        if int(r) != 0:
            return {"ok": False, "msg": f"MoveJ(A) err={r}"}
        
        # 2) DROP 위치로 이동 (Blocking - 완전히 도착 후 다음 동작)
        print(f"[PLACE] 2) DROP MoveCart (⚙️ 중간 속도, Blocking)")
        r = self.robot_motion.move_cart(
            pose6=drop,
            vel_list=cfg.MOVE_CART_VEL_PLACE,  # ⚙️ 중간 속도 (60, 40, 20)
            blendT=-1.0,  # 🔒 Blocking: 완전히 정지 후 다음 명령 실행
            label="DROP",
            reconnect_cb=reconnect_cb
        )
        if int(r) != 0:
            return {"ok": False, "msg": f"MoveCart(DROP) err={r}", "drop": drop}
        
        # 3) 그리퍼 열기 (놓기)
        print("[PLACE] 3) Gripper OPEN")
        try:
            self.gripper.open(reconnect_cb=reconnect_cb)
            
            # 그리퍼 안정화 대기 (박스가 완전히 놓이도록)
            import time
            settle_time = cfg.GRIPPER_SETTLE_TIME_OPEN
            print(f"[PLACE] 3-1) Gripper settle wait {settle_time}s...")
            time.sleep(settle_time)
        except Exception as e:
            return {"ok": False, "msg": f"gripper_open failed: {e}", "drop": drop}
        
        # 4) A 위치로 복귀
        r = self.robot_motion.move_j(
            joint_pos=a_joint,  # ✅ 수정: 현재 사용한 A 위치로 복귀 (위치1 또는 위치2)
            label="A back",
            reconnect_cb=reconnect_cb
        )
        if int(r) != 0:
            return {"ok": False, "msg": f"MoveJ(A back) err={r}"}
        
        # 5) HOME으로 복귀
        r = self.robot_motion.move_j(
            joint_pos=home_joint6,
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
    # 명령 (13번 기능 - Quick Start)
    # -------------------------
    def cmd13_quick_start(self, reconnect_cb: Optional[Callable] = None) -> Dict[str, Any]:
        """
        13번: Quick Start - 자동 초기화 후 4개 박스 이동
        0번 (연결) → 1번 (Vision ON) → 2번 (Home 저장) → 12번 (4개 자동)
        
        Args:
            reconnect_cb: 재연결 콜백
        
        Returns:
            결과 딕셔너리
        """
        print("\n" + "=" * 60)
        print("[13] Quick Start - 자동 초기화 및 4개 박스 이동")
        print("=" * 60)
        
        # 0) 로봇 연결
        if not self.connector.is_connected():
            print("[13] Step 0: 로봇 연결 중...")
            success = self.connector.connect()
            if not success:
                print("[13] ❌ 로봇 연결 실패")
                return {"ok": False, "msg": "robot connection failed"}
            print("[13] ✅ 로봇 연결 완료")
        else:
            print("[13] ✅ 로봇 이미 연결됨")
        
        # 1) Vision ON
        if not self.box_detector.is_running():
            print("[13] Step 1: Vision 시작 중...")
            success = self.box_detector.start()
            if not success:
                print("[13] ❌ Vision 시작 실패")
                return {"ok": False, "msg": "vision start failed"}
            print("[13] ✅ Vision 시작 완료")
            time.sleep(0.3)  # 카메라 안정화 대기 (1.0 → 0.3초)
        else:
            print("[13] ✅ Vision 이미 실행 중")
        
        # 2) Home 위치 저장
        print("[13] Step 2: 현재 위치를 Home으로 저장 중...")
        try:
            (e1, pose), (e2, joint) = self.robot_state.read_pose_joint(reconnect_cb=reconnect_cb)
            if e2 == 0 and joint is not None:
                self.robot_state.set_initial_joint6(joint)
                print(f"[13] ✅ Home 위치 저장 완료: {self.robot_state.fmt_joint(joint)}")
            else:
                print("[13] ❌ Home 위치 저장 실패")
                return {"ok": False, "msg": "home position save failed"}
        except Exception as e:
            print(f"[13] ❌ Home 위치 저장 중 에러: {e}")
            return {"ok": False, "msg": f"home save error: {e}"}
        
        # 3) 자동으로 4개 박스 이동
        print("\n[13] Step 3: 자동으로 4개 박스 이동 시작")
        print("=" * 60)
        
        # 스택 카운터 자동 리셋 (Quick Start는 항상 리셋)
        if self._stack_counter > 0:
            print(f"[13] 현재 스택 카운터: {self._stack_counter}")
            print("[13] Quick Start이므로 스택 카운터를 자동으로 0으로 리셋합니다.")
            self._stack_counter = 0
            print("[13] 스택 카운터 리셋 완료")
        
        # Home joint 가져오기
        home_joint6 = self.get_home_joint6(reconnect_cb=reconnect_cb)
        if home_joint6 is None:
            print("[13] ❌ home_joint6를 못 찾았음")
            return {"ok": False, "msg": "home_joint6 missing"}
        
        # Vision restart 허용으로 시작
        self.set_vision_allow_restart(True)
        
        # 4개 박스 자동 이동
        n = 4
        try:
            for i in range(n):
                print("\n" + "=" * 60)
                print(f"[13] Cycle {i+1}/{n}")
                print("=" * 60)
                
                # 0) HOME 정렬
                out_home0 = self.return_home.cmd_home_only(reconnect_cb=reconnect_cb)
                if not out_home0.get("ok", False):
                    print(f"[13] ❌ Cycle {i+1}: HOME(prepare) FAIL:", out_home0.get("msg", ""))
                    return {"ok": False, "msg": "home prepare fail", "cycle": i+1, "completed": i}
                
                # HOME 후 캐시 갱신 (cmd를 사용하여 캐시 저장)
                print(f"[13] Cycle {i+1}: 현재 위치 읽기...")
                result = self.robot_state.cmd_read_pose_joint(reconnect_cb=reconnect_cb)
                if result is None:
                    print(f"[13] ❌ Cycle {i+1}: 위치 읽기 실패")
                    return {"ok": False, "msg": "pose read failed", "cycle": i+1, "completed": i}
                print(f"[13] Cycle {i+1}: ✅ 위치 읽기 완료")
                
                # 3) Measure
                print(f"[13] Cycle {i+1}: 측정 중...")
                self.set_vision_allow_restart(True)
                meas = self.box_detector.cmd_measure_avg()
                # time.sleep(0.05)  제거 - 불필요
                
                if meas is None:
                    print(f"[13] ❌ Cycle {i+1}: 측정 실패")
                    return {"ok": False, "msg": "measure fail", "cycle": i+1, "completed": i}
                
                # 로봇 동작 중 restart 금지
                self.set_vision_allow_restart(False)
                
                # 4) Build target
                print(f"[13] Cycle {i+1}: 목표 계산 중...")
                self.target_pose.cmd_build_target_from_last(
                    use_last_pose=True,  # ⭐ True로 변경 (캐시 사용)
                    reconnect_cb=reconnect_cb
                )
                
                if self.target_pose.get_last_target_pose6() is None:
                    print(f"[13] ❌ Cycle {i+1}: 목표 계산 실패")
                    return {"ok": False, "msg": "target cache missing", "cycle": i+1, "completed": i}
                
                # 5) IK check
                print(f"[13] Cycle {i+1}: IK 검증 중...")
                self.ik_checker.cmd_check_target_from_last(
                    check_phase0=True,
                    reconnect_cb=reconnect_cb
                )
                
                if self.ik_checker.get_last_phase0_pose6() is None:
                    print(f"[13] ❌ Cycle {i+1}: IK 검증 실패")
                    return {"ok": False, "msg": "phase0 cache missing", "cycle": i+1, "completed": i}
                
                # 6) Smooth pick
                print(f"[13] Cycle {i+1}: Pick 중...")
                out_pick = self.pick_place.cmd6_smooth_auto(reconnect_cb=reconnect_cb)
                if not out_pick.get("ok", False):
                    print(f"[13] ❌ Cycle {i+1}: PICK FAIL:", out_pick.get("msg", ""))
                    return {"ok": False, "msg": "pick fail", "cycle": i+1, "completed": i, "detail": out_pick}
                
                # 6-1) 세로 박스 확인 및 처리
                is_vertical = False
                last_meas = self.box_detector.get_last_measure_avg()
                if last_meas is not None and last_meas.get("is_vertical", False):
                    is_vertical = True
                    
                    # 회전 기능 활성화 여부 확인
                    if bool(getattr(cfg, "ENABLE_VERTICAL_BOX_ROTATION", False)):
                        print(f"[13] 🔄 Cycle {i+1}: 세로 박스 회전 처리 시작")
                        
                        # 회전 처리
                        out_rotate = self.rotate_vertical_box_and_place(
                            home_joint6=home_joint6,
                            reconnect_cb=reconnect_cb
                        )
                        if not out_rotate.get("ok", False):
                            print(f"[13] ❌ Cycle {i+1}: ROTATE FAIL:", out_rotate.get("msg", ""))
                            return {"ok": False, "msg": "rotate fail", "cycle": i+1, "completed": i, "detail": out_rotate}
                        
                        # 재측정
                        self.set_vision_allow_restart(True)
                        meas2 = self.box_detector.cmd_measure_avg()
                        if meas2 is None:
                            print(f"[13] ❌ Cycle {i+1}: 재측정 실패")
                            return {"ok": False, "msg": "remeasure fail", "cycle": i+1, "completed": i}
                        
                        self.set_vision_allow_restart(False)
                        
                        # 재계산
                        self.target_pose.cmd_build_target_from_last(
                            use_last_pose=True,
                            reconnect_cb=reconnect_cb
                        )
                        self.ik_checker.cmd_check_target_from_last(
                            check_phase0=True,
                            reconnect_cb=reconnect_cb
                        )
                        
                        # 재잡기
                        out_pick2 = self.pick_place.cmd6_smooth_auto(reconnect_cb=reconnect_cb)
                        if not out_pick2.get("ok", False):
                            print(f"[13] ❌ Cycle {i+1}: RE-PICK FAIL:", out_pick2.get("msg", ""))
                            return {"ok": False, "msg": "re-pick fail", "cycle": i+1, "completed": i, "detail": out_pick2}
                
                # 7) HOME only (carry)
                print(f"[13] Cycle {i+1}: Home으로 이동 중...")
                out_home1 = self.return_home.cmd_home_only(reconnect_cb=reconnect_cb)
                if not out_home1.get("ok", False):
                    print(f"[13] ❌ Cycle {i+1}: HOME(carry) FAIL:", out_home1.get("msg", ""))
                    return {"ok": False, "msg": "home carry fail", "cycle": i+1, "completed": i}
                
                # 8) PLACE
                print(f"[13] Cycle {i+1}: Place 중...")
                out_place = self.place_one_stack(
                    home_joint6=home_joint6,
                    reconnect_cb=reconnect_cb
                )
                if not out_place.get("ok", False):
                    print(f"[13] ❌ Cycle {i+1}: PLACE FAIL:", out_place.get("msg", ""))
                    return {"ok": False, "msg": "place fail", "cycle": i+1, "completed": i, "detail": out_place}
                
                print(f"[13] ✅ Cycle {i+1}/{n} 완료! (stack_counter={self._stack_counter})")
                
                # 다음 사이클 전 restart 허용
                self.set_vision_allow_restart(True)
            
            print("\n" + "=" * 60)
            print(f"[13] 🎉 Quick Start 완료! 총 {n}개 박스 이동 완료")
            print(f"[13] 최종 스택 카운터: {self._stack_counter}")
            print("=" * 60)
            
            return {
                "ok": True,
                "msg": "quick start done",
                "count": n,
                "stack_counter": self._stack_counter,
                "completed": n
            }
        
        except Exception as e:
            print(f"[13] ❌ Exception: {e}")
            return {"ok": False, "msg": f"exception: {e}", "completed": 0}
        
        finally:
            # 어떤 종료 경로든 vision restart 허용 복구
            self.set_vision_allow_restart(True)
    
    # -------------------------
    # 명령 (12번 기능)
    # -------------------------
    def cmd12_auto_loop(self, num_cycles: int = None, reconnect_cb: Optional[Callable] = None) -> Dict[str, Any]:
        """
        12번: 자동 Pick & Place 루프
        N번 반복: 측정 → 계산 → IK → Pick → Home → Place
        
        안정화:
        - Step3(측정) 중에는 vision restart 허용
        - Step4~8(로봇 동작) 중에는 vision restart 금지 (크래시 방지)
        - 종료/에러/리턴 시 반드시 restart 허용으로 복구
        
        Args:
            num_cycles: 반복 횟수 (None이면 입력 받음)
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
        
        # 반복 횟수 (GUI에서 전달 또는 CLI 입력)
        if num_cycles is None:
            raw = input("박스 몇 개 옮길까요? (1~6개, b=back) > ").strip().lower()
            if raw in ("b", "back", "q", "quit"):
                return {"ok": True, "msg": "cancel"}
            
            try:
                n = int(raw)
                if n <= 0 or n > 6:
                    print("[12] 1~6 사이의 숫자를 입력하세요.")
                    return {"ok": False, "msg": "invalid count"}
            except Exception:
                print("[12] 숫자 입력이 아닙니다.")
                return {"ok": False, "msg": "invalid count"}
        else:
            n = num_cycles
            if n <= 0:
                print("[12] num_cycles는 1 이상이어야 합니다.")
                return {"ok": False, "msg": "invalid count"}
            if n > 6:
                print("[12] num_cycles는 최대 6까지 가능합니다.")
                return {"ok": False, "msg": "invalid count"}
        
        # Home joint 가져오기
        home_joint6 = self.get_home_joint6(reconnect_cb=reconnect_cb)
        if home_joint6 is None:
            print("[12] home_joint6를 못 찾았음. 2번(홈 저장)부터 하세요.")
            return {"ok": False, "msg": "home_joint6 missing"}
        
        # 스택 카운터는 GUI에서 "Reset Counter" 버튼으로 제어
        # 12번은 리셋하지 않고 계속 누적
        
        # 누적 카운터 체크 (최대 6개)
        if self._stack_counter >= 6:
            print(f"[12] ❌ 이미 {self._stack_counter}개 쌓았습니다. (최대 6개)")
            print(f"[12] Reset Counter 버튼을 눌러서 카운터를 리셋하세요.")
            return {"ok": False, "msg": "stack full (6개)", "stack_counter": self._stack_counter}
        
        # 이번 사이클로 6개 초과하는지 체크
        if self._stack_counter + n > 6:
            available = 6 - self._stack_counter
            print(f"[12] ⚠️ 현재 {self._stack_counter}개 쌓여있습니다.")
            print(f"[12] {n}개를 더 쌓으면 총 {self._stack_counter + n}개가 되어 6개를 초과합니다.")
            print(f"[12] 최대 {available}개까지만 가능합니다.")
            return {"ok": False, "msg": f"would exceed 6 (현재:{self._stack_counter}, 요청:{n})", "stack_counter": self._stack_counter}
        
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
                
                # HOME 후 캐시 갱신 (cmd를 사용하여 캐시 저장)
                print(f"[12] Cycle {i+1}: 현재 위치 읽기...")
                result = self.robot_state.cmd_read_pose_joint(reconnect_cb=reconnect_cb)
                if result is None:
                    print(f"[12] Cycle {i+1}: 위치 읽기 실패")
                    return {"ok": False, "msg": "read pose fail", "cycle": i+1}
                
                # 3) Measure (측정 중에는 restart 허용)
                print("[12] Step3: measure_avg")
                self.set_vision_allow_restart(True)
                meas = self.box_detector.cmd_measure_avg()
                # time.sleep(0.05)  제거 - 불필요
                
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
                
                # 6-1) 세로 박스 확인 및 처리
                is_vertical = False
                last_meas = self.box_detector.get_last_measure_avg()
                if last_meas is not None and last_meas.get("is_vertical", False):
                    is_vertical = True
                    
                    # 회전 기능 활성화 여부 확인
                    if bool(getattr(cfg, "ENABLE_VERTICAL_BOX_ROTATION", False)):
                        print("[12] 🔄 세로 박스 감지! 회전 처리 시작")
                        
                        # 회전 처리
                        out_rotate = self.rotate_vertical_box_and_place(
                            home_joint6=home_joint6,
                            reconnect_cb=reconnect_cb
                        )
                        if not out_rotate.get("ok", False):
                            print("[12] ROTATE FAIL:", out_rotate.get("msg", ""))
                            return {"ok": False, "msg": "rotate fail", "cycle": i+1, "detail": out_rotate}
                        
                        print("[12] 재측정 및 재잡기 시작")
                        # 재측정
                        self.set_vision_allow_restart(True)
                        meas2 = self.box_detector.cmd_measure_avg()
                        if meas2 is None:
                            print("[12] 재측정 실패")
                            return {"ok": False, "msg": "remeasure fail", "cycle": i+1}
                        
                        self.set_vision_allow_restart(False)
                        
                        # 재계산
                        self.target_pose.cmd_build_target_from_last(
                            use_last_pose=True,
                            reconnect_cb=reconnect_cb
                        )
                        self.ik_checker.cmd_check_target_from_last(
                            check_phase0=True,
                            reconnect_cb=reconnect_cb
                        )
                        
                        # 재잡기
                        out_pick2 = self.pick_place.cmd6_smooth_auto(reconnect_cb=reconnect_cb)
                        if not out_pick2.get("ok", False):
                            print("[12] RE-PICK FAIL:", out_pick2.get("msg", ""))
                            return {"ok": False, "msg": "re-pick fail", "cycle": i+1, "detail": out_pick2}
                    else:
                        print("[12] ⚠️  세로 박스 회전 처리 비활성화됨 (ENABLE_VERTICAL_BOX_ROTATION=False)")
                        print("[12] 세로 박스를 그대로 처리합니다 (감지만 하고 넘어감)")
                
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