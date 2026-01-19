# main_gui.py
"""
GUI 버전 실행
"""
import sys
import tkinter as tk

from config.app_config import ROBOT_IP_DEFAULT, FAIRINO_PYD_PATH
from robot import (
    RobotConnector, RobotState, RobotMotion, 
    GripperController, ReturnHome, TargetPose,
    IKChecker, PickPlace, AutoPickPlace, SafePickPlace
)
from vision import BoxDetector
from gui import RobotGUI


def main():
    """메인 함수"""
    # 컨트롤러 생성
    robot_connector = RobotConnector(
        ip=ROBOT_IP_DEFAULT,
        sdk_path=FAIRINO_PYD_PATH
    )
    
    robot_state = RobotState(robot_connector)
    robot_motion = RobotMotion(robot_connector, robot_state)
    gripper_controller = GripperController(robot_connector, robot_state)
    return_home = ReturnHome(robot_connector, robot_state, robot_motion, gripper_controller)
    
    # Vision 추가
    box_detector = BoxDetector()
    
    # Auto Pick & Place 추가
    target_pose = TargetPose(robot_state, box_detector)
    ik_checker = IKChecker(robot_connector, robot_state, target_pose)
    pick_place = PickPlace(robot_connector, robot_state, robot_motion, 
                          gripper_controller, ik_checker, target_pose)
    auto_pick_place = AutoPickPlace(
        robot_connector, robot_state, robot_motion, gripper_controller,
        box_detector, target_pose, ik_checker, pick_place, return_home
    )
    
    # Safe Pick & Place 추가 (Manual Stacking용)
    safe_pick_place = SafePickPlace(
        robot_connector, robot_state, robot_motion, gripper_controller
    )
    
    # 컨트롤러 딕셔너리
    controllers = {
        'robot_connector': robot_connector,
        'robot_state': robot_state,
        'robot_motion': robot_motion,
        'gripper_controller': gripper_controller,
        'return_home': return_home,
        'box_detector': box_detector,
        'auto_pick_place': auto_pick_place,
        'safe_pick_place': safe_pick_place,
    }
    
    # GUI 실행
    root = tk.Tk()
    app = RobotGUI(root, controllers)
    root.mainloop()


if __name__ == "__main__":
    main()
