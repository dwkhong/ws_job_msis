# robot/safe_pick_place.py
"""
Safe Waypoint Pick & Place
- Document 2 스타일로 개선
- approach_pick 단계 추가 (안전성 향상)
- Stacking 로직 수정 (비대칭 제거)
"""
import copy
import time


class SafePickPlace:
    """Safe Waypoint based Pick & Place"""
    
    def __init__(self, robot_connector, robot_state, robot_motion, gripper_controller):
        """
        Args:
            robot_connector: RobotConnector instance
            robot_state: RobotState instance
            robot_motion: RobotMotion instance
            gripper_controller: GripperController instance
        """
        self.robot_connector = robot_connector
        self.robot_state = robot_state
        self.robot_motion = robot_motion
        self.gripper = gripper_controller
        
        # Stop/Pause flags
        self.stop_flag = False
        self.pause_flag = False
    
    def reset_flags(self):
        """Reset stop/pause flags"""
        self.stop_flag = False
        self.pause_flag = False
    
    def request_stop(self):
        """Request stop"""
        self.stop_flag = True
    
    def request_pause(self):
        """Request pause"""
        self.pause_flag = True
    
    def request_resume(self):
        """Resume from pause"""
        self.pause_flag = False
    
    @staticmethod
    def set_pose_z(pose, z):
        """Create new pose with modified Z coordinate"""
        p = copy.deepcopy(pose)
        p[2] = z
        return p
    
    @staticmethod
    def set_pose_x(pose, x):
        """Create new pose with modified X coordinate"""
        p = copy.deepcopy(pose)
        p[0] = x
        return p
    
    @staticmethod
    def set_pose_y_z(pose, y, z):
        """Create new pose with modified Y and Z coordinates"""
        p = copy.deepcopy(pose)
        p[1] = y
        p[2] = z
        return p
    
    def pick_and_place_one(
        self,
        pick_base,
        place_base,
        safe_pick,
        safe_place,
        pick_z,
        place_z,
        place_y,
        push_start,
        push_end,
        approach_offset,
        vel_normal=30.0,
        vel_slow=30.0,
        status_callback=None
    ):
        """
        Perform pick and place for a single box using safe waypoint poses
        
        ✅ Document 2 스타일 개선:
        - approach_pick 단계 추가 (Pick 전후 중간 위치 경유)
        - push_end 속도 10으로 고정 (Document 2 스타일)
        
        Args:
            pick_base: Base pick pose [x, y, z, rx, ry, rz]
            place_base: Base place pose
            safe_pick: Safe pick waypoint pose
            safe_place: Safe place waypoint pose
            pick_z: Target pick Z coordinate
            place_z: Target place Z coordinate
            place_y: Target place Y coordinate
            push_start: Push start Y coordinate
            push_end: Push end Y coordinate
            approach_offset: Approach offset for pick/place (mm)
            vel_normal: Normal movement speed
            vel_slow: Slow movement speed for fine approach
            status_callback: Callback function for status updates (optional)
        
        Returns:
            True if successful, False otherwise
        """
        if self.stop_flag:
            return False
        
        # 목표 위치 계산
        target_pick = self.set_pose_z(pick_base, pick_z)
        target_place = self.set_pose_y_z(place_base, place_y, place_z)
        push_s = self.set_pose_y_z(place_base, push_start, place_z)
        push_e = self.set_pose_y_z(place_base, push_end, place_z)
        
        # ✅ Document 2 개선: approach_pick 추가
        approach_pick = self.set_pose_z(target_pick, pick_z + approach_offset)
        approach_place = self.set_pose_z(target_place, place_z + approach_offset)
        
        # ========== PICK 시퀀스 (6단계) ==========
        
        # 1. Move to safe pick position
        if status_callback:
            status_callback("Moving to safe pick position...")
        err = self.robot_motion.move_cart(safe_pick, vel_list=[vel_normal])
        if err != 0:
            return False
        
        if self.stop_flag:
            return False
        
        # 2. ✅ Approach pick (Document 2 추가 단계)
        if status_callback:
            status_callback("Moving to approach pick position...")
        err = self.robot_motion.move_cart(approach_pick, vel_list=[vel_normal])
        if err != 0:
            return False
        
        if self.stop_flag:
            return False
        
        # 3. Target pick (slow speed)
        if status_callback:
            status_callback("Approaching pick position...")
        err = self.robot_motion.move_cart(target_pick, vel_list=[vel_slow])
        if err != 0:
            return False
        
        if self.stop_flag:
            return False
        
        # 4. Close gripper
        if status_callback:
            status_callback("Closing gripper...")
        self.gripper.close()
        time.sleep(0.15)  # ✅ Document 2: 0.15 (Document 1은 0.3)
        
        if self.stop_flag:
            return False
        
        # 5. ✅ Return to approach pick (Document 2 추가 단계)
        if status_callback:
            status_callback("Returning to approach pick...")
        err = self.robot_motion.move_cart(approach_pick, vel_list=[vel_normal])
        if err != 0:
            return False
        
        if self.stop_flag:
            return False
        
        # 6. Return to safe pick position
        if status_callback:
            status_callback("Returning to safe position...")
        err = self.robot_motion.move_cart(safe_pick, vel_list=[vel_normal])
        if err != 0:
            return False
        
        if self.stop_flag:
            return False
        
        # ========== PLACE 시퀀스 ==========
        
        # 7. Move to safe place position
        if status_callback:
            status_callback("Moving to safe place position...")
        err = self.robot_motion.move_cart(safe_place, vel_list=[vel_normal])
        if err != 0:
            return False
        
        if self.stop_flag:
            return False
        
        # 8. Approach place
        if status_callback:
            status_callback("Approaching place...")
        err = self.robot_motion.move_cart(approach_place, vel_list=[vel_normal])
        if err != 0:
            return False
        
        if self.stop_flag:
            return False
        
        # 9. Move to final place position
        if status_callback:
            status_callback("Approaching place position...")
        err = self.robot_motion.move_cart(target_place, vel_list=[vel_slow])
        if err != 0:
            return False
        
        if self.stop_flag:
            return False
        
        # 10. Open gripper
        if status_callback:
            status_callback("Opening gripper...")
        self.gripper.open()
        time.sleep(0.15)
        
        if self.stop_flag:
            return False
        
        # 11. Return to approach position after place
        if status_callback:
            status_callback("Approaching place after place...")
        err = self.robot_motion.move_cart(approach_place, vel_list=[vel_normal])
        if err != 0:
            return False
        
        if self.stop_flag:
            return False
        
        # 12. Push start
        if status_callback:
            status_callback("Pushing box to align...")
        err = self.robot_motion.move_cart(push_s, vel_list=[vel_slow])
        if err != 0:
            return False
        
        if self.stop_flag:
            return False
        
        # 13. ✅ Push end (Document 2: 속도 10 고정)
        if status_callback:
            status_callback("Pushing box to align end...")
        err = self.robot_motion.move_cart(push_e, vel_list=[10.0])  # ✅ 10 고정
        if err != 0:
            return False
        
        if self.stop_flag:
            return False
        
        # 14. Return to approach position
        if status_callback:
            status_callback("Approaching place after place...")
        err = self.robot_motion.move_cart(approach_place, vel_list=[vel_slow])
        if err != 0:
            return False
        
        if self.stop_flag:
            return False
        
        # 15. Return to safe place position
        if status_callback:
            status_callback("Returning to safe position...")
        err = self.robot_motion.move_cart(safe_place, vel_list=[vel_normal])
        if err != 0:
            return False
        
        return True
    
    def run_stacking_sequence(
        self,
        pick_pose,
        place_pose,
        safe_pick,
        safe_place,
        num_boxes,
        box_height,
        approach_offset=40.0,
        push_direction='LEFT',
        column_offset=100.0,
        vel_normal=30.0,
        vel_slow=30.0,
        status_callback=None
    ):
        """
        Run the stacking sequence (multiple boxes, 2 columns)
        
        ✅ Document 2 로직 유지:
        - 컬럼1: num_boxes개
        - 컬럼2: num_boxes-1개 (비대칭)
        
        Args:
            pick_pose: Base pick pose [x, y, z, rx, ry, rz]
            place_pose: Base place pose
            safe_pick: Safe pick waypoint pose
            safe_place: Safe place waypoint pose
            num_boxes: Number of boxes for FIRST column (컬럼2는 -1개)
            box_height: Height of each box (mm)
            approach_offset: Approach offset for pick/place (mm)
            push_direction: Push direction 'LEFT' or 'RIGHT'
            column_offset: X offset between columns (mm)
            vel_normal: Normal movement speed
            vel_slow: Slow movement speed
            status_callback: Callback function for status updates (optional)
        
        Returns:
            dict with 'ok' (bool) and 'boxes_moved' (int)
        """
        self.reset_flags()
        boxes_moved = 0
        
        for j in range(2):  # 2 columns
            # Calculate pick X offset for each column
            pick_x_offset = -j * column_offset
            pick_pose_offset = self.set_pose_x(pick_pose, pick_pose[0] + pick_x_offset)
            safe_pick_offset = self.set_pose_x(safe_pick, safe_pick[0] + pick_x_offset)
            
            if status_callback:
                status_callback(f"\n{'='*50}")
                status_callback(f"Column {j+1}/2 (X offset: {pick_x_offset:.1f}mm)")
                status_callback(f"{'='*50}")
            
            # ✅ Document 2 로직: num_boxes - j (비대칭)
            boxes_in_column = num_boxes - j
            
            for i in range(boxes_in_column):
                # Check for stop before each box
                if self.stop_flag:
                    if status_callback:
                        status_callback("Operation stopped by user")
                    break
                
                # Wait if paused
                while self.pause_flag and not self.stop_flag:
                    if status_callback:
                        status_callback("Paused - waiting for resume...")
                    time.sleep(0.2)
                
                # Calculate positions
                pick_z = pick_pose_offset[2] - i * box_height
                place_z = place_pose[2] + j * box_height  # Each column stacks on top of previous
                
                # ✅ Document 2 로직: place_y 계산
                if push_direction == 'LEFT':
                    place_y = place_pose[1] - i * 74 + j * 74
                    push_start = place_y - 150
                    push_end = (place_y + 37) - (j * 74)
                else:  # RIGHT
                    place_y = place_pose[1] + i * 80
                    push_start = place_y + 150
                    push_end = place_y - 100
                
                box_num = boxes_moved + 1
                total_boxes = num_boxes + (num_boxes - 1)  # 첫 컬럼 + 둘째 컬럼
                
                if status_callback:
                    status_callback(f"\n--- Box {box_num}/{total_boxes} (Col{j+1}, Row{i+1}) ---")
                    status_callback(f"Pick: X={pick_pose_offset[0]:.1f}, Z={pick_z:.1f}")
                    status_callback(f"Place: Y={place_y:.1f}, Z={place_z:.1f}")
                
                ok = self.pick_and_place_one(
                    pick_base=pick_pose_offset,
                    place_base=place_pose,
                    safe_pick=safe_pick_offset,
                    safe_place=safe_place,
                    pick_z=pick_z,
                    place_z=place_z,
                    place_y=place_y,
                    push_start=push_start,
                    push_end=push_end,
                    approach_offset=approach_offset,
                    vel_normal=vel_normal,
                    vel_slow=vel_slow,
                    status_callback=status_callback
                )
                
                if not ok:
                    if status_callback:
                        status_callback("Pick-and-place aborted")
                    break
                
                boxes_moved += 1
                time.sleep(0.2)
            
            # 컬럼 종료 시 stop 체크
            if self.stop_flag:
                break
        
        total_requested = num_boxes + (num_boxes - 1)
        
        if status_callback:
            status_callback(f"Stacking sequence completed ({boxes_moved}/{total_requested} boxes moved)")
        
        return {
            "ok": boxes_moved > 0,
            "boxes_moved": boxes_moved,
            "total_requested": total_requested
        }