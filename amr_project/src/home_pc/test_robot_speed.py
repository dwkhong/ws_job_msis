#!/usr/bin/env python3
"""
로봇 속도 테스트 스크립트
MoveCart를 이용해 세 지점을 최고속도로 왕복
"""
import time
import sys
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
WP11_DROP_BASE_POSE = [75.776, -445.201, 241.223, 178.430, -1.496, 88.630]
CAM_POSE = [329.772, -101.955, 802.907, -179.919, 0.279, 179.345]

# ============================================================
# 속도 설정 (여러 가지 테스트)
# ============================================================
SPEED_TESTS = [
    {"name": "최고속 (vel=100, ovl=100)", "vel": 100.0, "ovl": 100.0},
    {"name": "고속 (vel=80, ovl=100)", "vel": 80.0, "ovl": 100.0},
    {"name": "중속 (vel=60, ovl=100)", "vel": 60.0, "ovl": 100.0},
    {"name": "저속 (vel=40, ovl=100)", "vel": 40.0, "ovl": 100.0},
    {"name": "최저속 (vel=20, ovl=100)", "vel": 20.0, "ovl": 100.0},
]

def main():
    """메인 함수"""
    print("=" * 60)
    print("로봇 속도 테스트")
    print("=" * 60)
    
    # 로봇 연결
    print("\n[1] 로봇 연결 중...")
    print(f"    IP: {ROBOT_IP_DEFAULT}")
    print(f"    SDK: {FAIRINO_PYD_PATH}")
    
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
        ("WP11_A", WP11_A_POSE),  # 복귀
    ]
    
    print("\n[3] 경로:")
    for i, (name, pose) in enumerate(route, 1):
        print(f"    {i}. {name}: [{pose[0]:.1f}, {pose[1]:.1f}, {pose[2]:.1f}]")
    
    # 사용자 선택
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
        # 모든 속도 테스트
        print("\n" + "=" * 60)
        print("🚀 모든 속도 순차 테스트 시작!")
        print("=" * 60)
        
        for test in SPEED_TESTS:
            run_speed_test(robot_motion, route, test)
            print("\n다음 테스트까지 3초 대기...")
            time.sleep(3)
    else:
        # 선택한 속도 테스트
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(SPEED_TESTS):
                test = SPEED_TESTS[idx]
                print("\n" + "=" * 60)
                print(f"🚀 {test['name']} 테스트 시작!")
                print("=" * 60)
                run_speed_test(robot_motion, route, test)
            else:
                print("❌ 잘못된 선택입니다.")
        except ValueError:
            print("❌ 잘못된 입력입니다.")
    
    print("\n" + "=" * 60)
    print("✅ 테스트 완료!")
    print("=" * 60)

def run_speed_test(robot_motion, route, test):
    """속도 테스트 실행"""
    vel = test["vel"]
    ovl = test["ovl"]
    
    print(f"\n[테스트] {test['name']}")
    print(f"설정: vel={vel}, acc=0.0, ovl={ovl}")
    print("-" * 60)
    
    total_time = 0.0
    
    for i, (name, pose) in enumerate(route, 1):
        print(f"\n{i}/{len(route)} → {name} 이동 중...", end=" ", flush=True)
        
        start_time = time.time()
        
        # MoveCart 실행
        rtn = robot_motion.move_cart(
            pose6=pose,
            vel_list=[vel],  # 단일 속도만
            acc=0.0,
            ovl=ovl,
            blendT=100,
            config=-1,
            label=name
        )
        
        elapsed = time.time() - start_time
        total_time += elapsed
        
        if int(rtn) == 0:
            print(f"✅ 성공 ({elapsed:.2f}초)")
        else:
            print(f"❌ 실패 (err={rtn}, {elapsed:.2f}초)")
    
    print("-" * 60)
    print(f"총 소요 시간: {total_time:.2f}초")
    print(f"평균 이동 시간: {total_time/len(route):.2f}초")
    
    # 사이클 타임 추정
    cycle_time = total_time / len(route) * 4  # Pick + Place + Return 등
    print(f"\n📊 예상 사이클 타임: {cycle_time:.2f}초/박스")
    print(f"📊 예상 시간당 생산: {3600/cycle_time:.0f}박스/시간")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n사용자 중단")
    except Exception as e:
        print(f"\n\n❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()