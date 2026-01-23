#!/usr/bin/env python3
"""
로봇 속도 테스트 스크립트 (실시간 모니터링)
- MoveCart 실행 중 실시간 속도 확인
- 명령 속도 vs 실제 속도 비교
- 상세 로그 출력
"""
import time
import sys
import threading
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from robot.robot_connector import RobotConnector
from robot.robot_state import RobotState
from robot.robot_motion import RobotMotion
from config.app_config import ROBOT_IP_DEFAULT, FAIRINO_PYD_PATH

# ============================================================
# 테스트 포즈
# ============================================================
WP11_A_POSE = [84.135, -445.200, 714.079, 179.285, 0.288, 87.147]
WP11_DROP_BASE_POSE = [75.776, -445.201, 400.223, 178.430, -1.496, 88.630]
CAM_POSE = [329.772, -101.955, 802.907, -179.919, 0.279, 179.345]

# ============================================================
# 속도 설정
# ============================================================
SPEED_TESTS = [
    {"name": "최고속 (vel=100)", "vel": 100.0, "ovl": 100.0},
    {"name": "고속 (vel=80)", "vel": 80.0, "ovl": 100.0},
    {"name": "중속 (vel=60)", "vel": 60.0, "ovl": 100.0},
    {"name": "저속 (vel=40)", "vel": 40.0, "ovl": 100.0},
    {"name": "최저속 (vel=20)", "vel": 20.0, "ovl": 100.0},
]


class SpeedMonitor:
    """실시간 속도 모니터 (백그라운드 스레드)"""
    
    def __init__(self, robot):
        self.robot = robot
        self.running = False
        self.thread = None
        self.speeds = []  # 속도 기록
        self.max_speed = 0.0
        self.avg_speed = 0.0
    
    def start(self):
        """모니터링 시작"""
        self.running = True
        self.speeds = []
        self.max_speed = 0.0
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """모니터링 중지"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        
        # 통계 계산
        if self.speeds:
            self.max_speed = max(self.speeds)
            self.avg_speed = sum(self.speeds) / len(self.speeds)
    
    def _monitor_loop(self):
        """모니터링 루프 (백그라운드)"""
        while self.running:
            try:
                # TCP 피드백 속도 조회
                err, speed = self.robot.GetActualTCPSpeed(flag=1)
                if err == 0 and speed:
                    # 선형 속도 계산 (mm/s)
                    linear_speed = (speed[0]**2 + speed[1]**2 + speed[2]**2)**0.5
                    self.speeds.append(linear_speed)
                
                time.sleep(0.1)  # 100ms 간격
            except Exception:
                pass
    
    def get_summary(self):
        """통계 요약"""
        return {
            "samples": len(self.speeds),
            "max_speed": self.max_speed,
            "avg_speed": self.avg_speed,
            "speeds": self.speeds
        }


def run_speed_test_with_monitor(robot_motion, robot, route, test):
    """속도 테스트 실행 (실시간 모니터링)"""
    vel = test["vel"]
    ovl = test["ovl"]
    
    print(f"\n{'='*60}")
    print(f"[테스트] {test['name']}")
    print(f"{'='*60}")
    print(f"📊 설정: vel={vel}, ovl={ovl}")
    print(f"📊 예상 최대 속도: ~{vel * ovl / 100:.1f}%")
    print("-" * 60)
    
    total_time = 0.0
    all_stats = []
    
    for i, (name, pose) in enumerate(route, 1):
        print(f"\n[{i}/{len(route)}] → {name} 이동")
        print(f"   목표: [{pose[0]:.1f}, {pose[1]:.1f}, {pose[2]:.1f}]")
        
        # 속도 모니터 시작
        monitor = SpeedMonitor(robot)
        monitor.start()
        
        start_time = time.time()
        
        # MoveCart 실행
        rtn = robot_motion.move_cart(
            pose6=pose,
            vel_list=[vel],
            acc=0.0,
            ovl=ovl,
            blendT=100,
            config=-1,
            label=name
        )
        
        elapsed = time.time() - start_time
        
        # 속도 모니터 중지
        monitor.stop()
        stats = monitor.get_summary()
        
        total_time += elapsed
        
        if int(rtn) == 0:
            print(f"   ✅ 성공 ({elapsed:.2f}초)")
            
            # 속도 통계 출력
            if stats["samples"] > 0:
                print(f"   📊 최대 속도: {stats['max_speed']:.2f} mm/s")
                print(f"   📊 평균 속도: {stats['avg_speed']:.2f} mm/s")
                print(f"   📊 샘플 수: {stats['samples']}개")
                all_stats.append(stats)
            
            # 현재 위치 확인
            try:
                err_pose, cur_pose = robot.GetActualTCPPose(flag=1)
                if err_pose == 0:
                    print(f"   📍 도착: [{cur_pose[0]:.1f}, {cur_pose[1]:.1f}, {cur_pose[2]:.1f}]")
            except Exception:
                pass
        else:
            print(f"   ❌ 실패 (err={rtn}, {elapsed:.2f}초)")
    
    # 전체 요약
    print("\n" + "=" * 60)
    print("📊 전체 요약")
    print("=" * 60)
    print(f"총 소요 시간: {total_time:.2f}초")
    print(f"평균 이동 시간: {total_time/len(route):.2f}초/이동")
    
    if all_stats:
        overall_max = max(s["max_speed"] for s in all_stats)
        overall_avg = sum(s["avg_speed"] for s in all_stats) / len(all_stats)
        print(f"전체 최대 속도: {overall_max:.2f} mm/s")
        print(f"전체 평균 속도: {overall_avg:.2f} mm/s")
    
    # 사이클 타임 추정
    cycle_time = total_time / len(route) * 4
    print(f"\n📊 예상 사이클 타임: {cycle_time:.2f}초/박스")
    print(f"📊 예상 시간당 생산: {3600/cycle_time:.0f}박스/시간")


def main():
    """메인 함수"""
    print("=" * 60)
    print("🚀 로봇 속도 테스트 (실시간 모니터링)")
    print("=" * 60)
    
    # 로봇 연결
    print("\n[1] 로봇 연결 중...")
    print(f"    IP: {ROBOT_IP_DEFAULT}")
    
    connector = RobotConnector(
        ip=ROBOT_IP_DEFAULT,
        sdk_path=FAIRINO_PYD_PATH
    )
    if not connector.connect():
        print("❌ 로봇 연결 실패!")
        return
    print("✅ 로봇 연결 성공")
    
    robot_state = RobotState(connector)
    robot_motion = RobotMotion(connector, robot_state)
    robot = connector.get_robot()
    
    # 현재 위치 읽기
    print("\n[2] 현재 위치 확인...")
    (err_p, pose), (err_j, joint) = robot_state.read_pose_joint()
    if err_p != 0 or pose is None:
        print(f"❌ 현재 위치 읽기 실패! err={err_p}")
        return
    print(f"✅ 현재 위치: [{pose[0]:.1f}, {pose[1]:.1f}, {pose[2]:.1f}]")
    
    # 경로 설정
    route = [
        ("WP11_A", WP11_A_POSE),
        ("DROP", WP11_DROP_BASE_POSE),
        ("CAM", CAM_POSE),
        ("WP11_A", WP11_A_POSE),
    ]
    
    print("\n[3] 경로:")
    for i, (name, pose) in enumerate(route, 1):
        print(f"    {i}. {name}: [{pose[0]:.1f}, {pose[1]:.1f}, {pose[2]:.1f}]")
    
    # 테스트 선택
    print("\n" + "=" * 60)
    print("테스트 모드 선택:")
    print("=" * 60)
    for i, test in enumerate(SPEED_TESTS, 1):
        print(f"{i}. {test['name']}")
    print("0. 모든 속도 순차 테스트")
    print("q. 종료")
    
    choice = input("\n선택 (0-5 또는 q): ").strip()
    
    if choice.lower() == 'q':
        print("종료합니다.")
        return
    
    # 테스트 실행
    if choice == '0':
        print("\n" + "=" * 60)
        print("🚀 모든 속도 순차 테스트 시작!")
        print("=" * 60)
        
        for test in SPEED_TESTS:
            run_speed_test_with_monitor(robot_motion, robot, route, test)
            print("\n⏳ 다음 테스트까지 3초 대기...")
            time.sleep(3)
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(SPEED_TESTS):
                test = SPEED_TESTS[idx]
                run_speed_test_with_monitor(robot_motion, robot, route, test)
            else:
                print("❌ 잘못된 선택입니다.")
        except ValueError:
            print("❌ 잘못된 입력입니다.")
    
    print("\n" + "=" * 60)
    print("✅ 테스트 완료!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자 중단")
    except Exception as e:
        print(f"\n\n❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()