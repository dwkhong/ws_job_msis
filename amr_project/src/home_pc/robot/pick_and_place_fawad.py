# robot/pick_and_place_fawad.py
"""
Safe Waypoint Pick & Place
- Original GUI stacking logic (safe pick → pick → safe pick → safe place → place → safe place)
- Uses saved poses from pose_manager
"""
import copy
import time


class SafePickPlace:
    """Safe Waypoint based Pick & Place (Original GUI logic)"""
    
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
    
    def pick_and_place_one(
        self,
        pick_base,
        place_base,
        safe_pick,
        safe_place,
        pick_z,
        place_z,
        vel_normal=30.0,
        vel_slow=30.0,
        status_callback=None
    ):
        """
        Perform pick and place for a single box using safe waypoint poses
        
        Args:
            pick_base: Base pick pose [x, y, z, rx, ry, rz]
            place_base: Base place pose
            safe_pick: Safe pick waypoint pose
            safe_place: Safe place waypoint pose
            pick_z: Target pick Z coordinate
            place_z: Target place Z coordinate
            vel_normal: Normal movement speed
            vel_slow: Slow movement speed for fine approach
            status_callback: Callback function for status updates (optional)
        
        Returns:
            True if successful, False otherwise
        """
        if self.stop_flag:
            return False
        
        target_pick = self.set_pose_z(pick_base, pick_z)
        target_place = self.set_pose_z(place_base, place_z)
        
        # 1. Move to safe pick position
        if status_callback:
            status_callback("Moving to safe pick position...")
        err = self.robot_motion.move_cart(safe_pick, vel=vel_normal)
        if err != 0:
            if status_callback:
                status_callback("Failed to move to safe pick")
            return False
        
        if self.stop_flag:
            return False
        
        # 2. Approach and pick (slow speed)
        if status_callback:
            status_callback("Approaching pick position...")
        err = self.robot_motion.move_cart(target_pick, vel=vel_slow)
        if err != 0:
            if status_callback:
                status_callback("Failed to approach pick")
            return False
        
        if self.stop_flag:
            return False
        
        # 3. Close gripper
        if status_callback:
            status_callback("Closing gripper...")
        self.gripper.close()
        time.sleep(0.15)
        
        if self.stop_flag:
            return False
        
        # 4. Return to safe pick position
        if status_callback:
            status_callback("Returning to safe pick...")
        err = self.robot_motion.move_cart(safe_pick, vel=vel_normal)
        if err != 0:
            if status_callback:
                status_callback("Failed to return to safe pick")
            return False
        
        if self.stop_flag:
            return False
        
        # 5. Move to safe place position
        if status_callback:
            status_callback("Moving to safe place position...")
        err = self.robot_motion.move_cart(safe_place, vel=vel_normal)
        if err != 0:
            if status_callback:
                status_callback("Failed to move to safe place")
            return False
        
        if self.stop_flag:
            return False
        
        # 6. Approach and place
        if status_callback:
            status_callback("Approaching place position...")
        err = self.robot_motion.move_cart(target_place, vel=vel_normal)
        if err != 0:
            if status_callback:
                status_callback("Failed to approach place")
            return False
        
        if self.stop_flag:
            return False
        
        # 7. Open gripper
        if status_callback:
            status_callback("Opening gripper...")
        self.gripper.open()
        time.sleep(0.15)
        
        if self.stop_flag:
            return False
        
        # 8. Return to safe place position
        if status_callback:
            status_callback("Returning to safe place...")
        err = self.robot_motion.move_cart(safe_place, vel=vel_normal)
        if err != 0:
            if status_callback:
                status_callback("Failed to return to safe place")
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
        vel_normal=30.0,
        vel_slow=30.0,
        status_callback=None
    ):
        """
        Run the stacking sequence (multiple boxes)
        
        Args:
            pick_pose: Base pick pose [x, y, z, rx, ry, rz]
            place_pose: Base place pose
            safe_pick: Safe pick waypoint pose
            safe_place: Safe place waypoint pose
            num_boxes: Number of boxes to stack
            box_height: Height of each box (mm)
            vel_normal: Normal movement speed
            vel_slow: Slow movement speed
            status_callback: Callback function for status updates (optional)
        
        Returns:
            dict with 'ok' (bool) and 'boxes_moved' (int)
        """
        self.reset_flags()
        boxes_moved = 0
        
        for i in range(num_boxes):
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
            
            # Calculate Z positions (pick from top, place on top)
            pick_z = pick_pose[2] - i * box_height
            place_z = place_pose[2] + i * box_height
            
            if status_callback:
                status_callback(f"\n--- Box {i+1}/{num_boxes} ---")
                status_callback(f"Pick Z: {pick_z:.3f}  Place Z: {place_z:.3f}")
            
            ok = self.pick_and_place_one(
                pick_base=pick_pose,
                place_base=place_pose,
                safe_pick=safe_pick,
                safe_place=safe_place,
                pick_z=pick_z,
                place_z=place_z,
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
        
        if status_callback:
            status_callback(f"Stacking sequence completed ({boxes_moved}/{num_boxes} boxes moved)")
        
        return {
            "ok": boxes_moved > 0,
            "boxes_moved": boxes_moved,
            "total_requested": num_boxes
        }