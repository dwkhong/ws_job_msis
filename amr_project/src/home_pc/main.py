# main.py
"""
Jetson Robot Vision 메인 프로그램
카메라를 이용한 박스 탐지 및 로봇팔 제어
"""
from config.app_config import ROBOT_IP_DEFAULT, FAIRINO_PYD_PATH
from robot import RobotConnector, RobotState, TargetPose, IKChecker
from vision import BoxDetector


def print_menu():
    """메인 메뉴 출력"""
    print("\n===== Main Menu =====")
    print("0. Robot Connect/Disconnect")
    print("1. Vision On/Off")
    print("2. Read Robot Pose/Joint (cache)")
    print("3. Vision Measure Avg (cache)")
    print("4. Build Target Pose (use last 2+3)")
    print("5. IK Check (use last 4 + last joint)")
    print("6. Smooth move 1-step ")
    print("7. Smooth move 2-step ")
    print("8. Return Home + Gripper Open")
    print("9. Gripper Control Menu")
    print("10. J4 Rotate (input delta deg)")
    print("11. J6 Rotate (input delta deg)")
    print("12. Auto Pick&Place (N cycles)")
    print("q. Quit")


def main():
    """메인 실행 함수"""
    # RobotConnector 인스턴스 생성
    robot_connector = RobotConnector(
        ip=ROBOT_IP_DEFAULT,
        sdk_path=FAIRINO_PYD_PATH
    )
    
    # RobotState 인스턴스 생성
    robot_state = RobotState(robot_connector)
    
    # BoxDetector 인스턴스 생성
    box_detector = BoxDetector()
    
    # TargetPose 인스턴스 생성
    target_pose = TargetPose(robot_state, box_detector)
    
    # IKChecker 인스턴스 생성
    ik_checker = IKChecker(robot_connector, robot_state, target_pose)
    
    print("🤖 Jetson Robot Vision System Started")
    print(f"📍 Robot IP: {ROBOT_IP_DEFAULT}")
    
    while True:
        print_menu()
        cmd = input("입력 (0~12/q) > ").strip().lower()
        
        if cmd == "q":
            print("\n👋 Shutting down...")
            # 종료 시 연결 해제
            if robot_connector.is_connected():
                robot_connector.disconnect()
            if box_detector.is_running():
                box_detector.stop()
            break
        
        # 0번: 로봇 연결/연결 해제
        elif cmd == "0":
            robot_connector.toggle()
        
        # 1번: Vision On/Off
        elif cmd == "1":
            box_detector.toggle()
        
        # 2번: 로봇 현재 위치 읽기
        elif cmd == "2":
            robot_state.cmd_read_pose_joint(reconnect_cb=lambda: robot_connector.connect())
        
        # 3번: Vision 평균 측정
        elif cmd == "3":
            box_detector.cmd_measure_avg()
        
        # 4번: 목표 Pose 계산
        elif cmd == "4":
            target_pose.cmd_build_target_from_last(
                use_last_pose=True,
                reconnect_cb=lambda: robot_connector.connect()
            )
        
        # 5번: IK 검증
        elif cmd == "5":
            ik_checker.cmd_check_target_from_last(
                check_phase0=True,
                reconnect_cb=lambda: robot_connector.connect()
            )
        
        # 4번: 목표 Pose 계산
        elif cmd == "4":
            target_pose.cmd_build_target_from_last(
                use_last_pose=True,
                reconnect_cb=lambda: robot_connector.connect()
            )
        
        # 5번: IK 검증
        elif cmd == "5":
            ik_checker.cmd_check_target_from_last(
                check_phase0=True,
                reconnect_cb=lambda: robot_connector.connect()
            )
        
        # 6~12번: 아직 미구현
        elif cmd in ["6", "7", "8", "9", "10", "11", "12"]:
            print(f"[INFO] 기능 {cmd}번은 아직 구현되지 않았습니다.")
        
        else:
            print("[ERROR] 잘못된 입력입니다.")


if __name__ == "__main__":
    main()