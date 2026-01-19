import math, ast
from .rest_api import PLAIN_API
from .util import compute_target_pose

import matplotlib.pyplot as plt
import numpy as np

class MSIS_PHOEBUS:
    def __init__(self, ip="", port=80):
        """
        Initializes the AMR motion client with a REST API endpoint.
        args:
         - ip: string IP address of the AMR controller.
         - port: integer TCP port for the AMR controller.
        return:
         - None: constructs the client and stores the API instance.
        """
        self.api = PLAIN_API(ip, port)

    def _update_target(self, ip):
        """Helper to update API target IP if provided."""
        if ip:
            self.api.ip = ip # Assuming PLAIN_API allows IP updates

    def shutdown_amr(self, ip=None):
        """
        Sends a shutdown command to the AMR for immediate power off.
        args:
         - None
        return:
         - dict or None: API response payload or None on error.
        """
        self._update_target(ip)
        payload = {
            "shutdown_time_interval": 0,
            "restart_time_interval": 0
        }
        return self.api._post("/api/core/system/v1/power/:shutdown", payload)
    
    def hibernate_amr(self, ip=None):
        """
        Puts the AMR into hibernate mode via the system API.
        args:
         - None
        return:
         - dict or None: API response payload or None on error.
        """
        self._update_target(ip)
        return self.api._post("/api/core/system/v1/power/:hibernate")
    
    def wakeup_amr(self, ip=None):
        """
        Wakes the AMR from hibernation using the system API endpoint.
        args:
         - None
        return:
         - dict or None: API response payload or None on error.
        """
        self._update_target(ip)
        return self.api._post("/api/core/system/v1/power/:wakeup")
    
    def restart_amr(self, ip=None):
        """
        Restarts the AMR modules through the system restart endpoint.
        args:
         - None
        return:
         - dict or None: API response payload or None on error.
        """
        self._update_target(ip)
        return self.api._post("/api/core/system/v1/power/:restartmodule")

    def get_power_status(self, _print=False, ip=None):
        """
        Retrieves the current power and battery status of the AMR.
        args:
         - _print: bool flag to print the response for debugging.
        return:
         - dict or None: power status information or None on error.
        """
        self._update_target(ip)
        response = self.api._get("/api/core/system/v1/power/status")
        if _print: print(response)
        return response

    def get_robot_speed(self, _print=False, ip=None):
        """
        Fetches the current robot speed vector from the motion API.
        args:
         - _print: bool flag to print the response for debugging.
        return:
         - dict or None: speed data including components or None on error.
        """
        self._update_target(ip)
        response = self.api._get("/api/core/motion/v1/speed")
        if _print: print(response)
        return response

    def get_robot_pose(self, _print=False, ip=None):
        """
        Obtains the current robot pose (x, y, yaw) from localization.
        args:
         - _print: bool flag to print the response for debugging.
        return:
         - dict or None: pose information or None on error.
        """
        self._update_target(ip)
        response = self.api._get("/api/core/slam/v1/localization/pose")
        if _print: print(response)
        return response

    def get_action_status(self, action_id, _print=False, ip=None):
        """
        Queries the status of a specific motion action by identifier.
        args:
         - action_id: identifier string of the action to query.
         - _print: bool flag to print the response for debugging.
        return:
         - dict or None: action status details or None on error.
        """
        self._update_target(ip)
        response = self.api._get(f"/api/core/motion/v1/actions/{action_id}")
        if _print: print(response)
        return response

    def stop_motion_now(self, ip=None):
        """
        Immediately stops the current motion action if one is running.
        args:
         - None
        return:
         - dict or None: API response indicating stop result or None on error.
        """
        self._update_target(ip)
        return self.api._delete("/api/core/motion/v1/actions/:current")
    
    def set_charge_station(self, ip=None):
        """
        Registers the current location as the designated charge station.
        args:
         - None
        return:
         - dict or None: API response payload or None on error.
        """
        self._update_target(ip)
        payload = {
            "metadata": {
                "display_name": "string"
            }
        }
        return self.api._post("/api/core/slam/v1/homedocks/:register", payload)
    
    def set_speed_limit(self, speed_mps, ip=None):
        """
        Dynamically adjusts the robot's maximum moving speed limit. 
        Adjusting this during a 'MoveTo' action allows for smooth 
        deceleration without aborting the current motion.
        
        args:
         - speed_mps: float target maximum speed in meters per second (m/s).
        return:
         - dict or None: API response payload or None on error.
        """
        self._update_target(ip)
        payload = {
            "param": "base.max_moving_speed",
            "value": str(speed_mps)
        }
        return self.api._put("/api/core/system/v1/parameter", payload)
    
    def go_to(self, target_x, target_y, yaw=None, speed_ratio=0.4, label="go_to", ip=None): 
        """
        Commands the AMR to move to a target coordinate with optional yaw.
        args:
         - target_x: float X-axis target position (meters).
         - target_y: float Y-axis target position (meters).
         - yaw: float target yaw in degrees or None to omit orientation.
         - label: string label for action tagging purposes.
        return:
         - None: posts a motion action, no direct return value.
        """
        self._update_target(ip)
        if yaw is None:  
            payload = {
                "action_name": "slamtec.agent.actions.MoveToAction",
                "options": {
                    "target": {"x": target_x, "y": target_y, "z": 0},
                    "move_options": {
                        "mode": 0,
                        "acceptable_precision": 0.01,
                        "fail_retry_count": 0,
                        "flags": ["precise"],
                        "speed_ratio": speed_ratio,
                    },
                },
            }
        else:
            payload = {
                "action_name": "slamtec.agent.actions.MoveToAction",
                "options": {
                    "target": {"x": target_x, "y": target_y, "z": 0},
                    "move_options": {
                        "mode": 0,
                        "acceptable_precision": 0.0001,
                        "fail_retry_count": 0,
                        "speed_ratio": speed_ratio,
                        "flags": ["precise","with_yaw"],
                        "yaw": math.radians(yaw)
                    },
                },
            }
            
        action = self.api._post("/api/core/motion/v1/actions", payload)
        return action

    def go_by_distance(self, angle, distance, label="go_by_distance", ip=None):
        """
        Moves the AMR by a distance at a given relative angle from current pose.
        args:
         - angle: float relative heading in degrees from current yaw.
         - distance: float distance to travel in meters.
         - label: string label for action tagging purposes.
        return:
         - None: submits a MoveTo action based on computed target pose.
        """
        self._update_target(ip)
        current_pose = self.get_robot_pose()
        target_pose = compute_target_pose(angle, current_pose, distance)

        payload = {
            "action_name": "slamtec.agent.actions.MoveToAction",
            "options": {
                "target": {"x": target_pose['x'], "y": target_pose['y'], "z": 0},
                "move_options": {
                    "mode": 0,
                    "acceptable_precision": 0.1,
                    "fail_retry_count": 0,
                    "speed_ratio": 1.0,
                },
            },
        }
        self.api._post("/api/core/motion/v1/actions", payload)
    
    def rotate_by_inPlace(self, deg=0, ip=None):
        """
        Rotates the AMR in place by a relative angle in degrees.
        args:
         - deg: float relative rotation angle in degrees.
        return:
         - None: posts a rotation action, no direct return value.
        """
        self._update_target(ip)
        payload = {"action_name": "slamtec.agent.actions.RotateAction", 
            "options": {
                "angle": math.radians(deg)
            }
        }

        self.api._post("/api/core/motion/v1/actions", payload)

    def rotate_to_inPlace(self, deg=0, unit='deg', ip=None):
        """
        Rotates the AMR in place to an absolute orientation.
        args:
         - deg: float target orientation value (degrees or radians).
         - unit: string specifying 'deg' for degrees or 'rad' for radians.
        return:
         - None: posts a rotate-to action, no direct return value.
        """
        self._update_target(ip)
        payload = {"action_name": "slamtec.agent.actions.RotateToAction", 
            "options": {
                "angle": math.radians(deg) if unit=='deg' else deg
            }
        }

        self.api._post("/api/core/motion/v1/actions", payload)

    def go_charge(self, ip=None):
        """
        Commands the AMR to return to and dock at the charge station.
        args:
         - None
        return:
         - None: posts a go-home docking action, no direct return value.
        """
        self._update_target(ip)
        payload = {
            "action_name": "slamtec.agent.actions.GoHomeAction",
            "options": {
                "flags": "dock",
                "back_to_landing": True,
                "charging_retry_count": 2,
                "move_options": {"mode": 0},
            },
        }
        self.api._post("/api/core/motion/v1/actions", payload)

    def get_laserscan(self, ip=None):
        self._update_target(ip)
        response = self.api._get("/api/core/system/v1/laserscan")
        return response.get("laser_points", [])
    
    # def go_rectilinear_by_distance(self, angle, distance):
    #     current_pose = self.get_robot_pose()
    #     target_pose = compute_target_pose(angle, current_pose, distance)

    #     # 1. Move to Target X (Maintain current Y)
    #     print(f"Moving to X: {target_pose['x']}")
    #     self.go_to(
    #         target_x=target_pose['x'], 
    #         target_y=current_pose['y'], 
    #         yaw=None # Don't worry about orientation yet
    #     )
        
    #     # 2. Move to Target Y (Reach final destination)
    #     print(f"Moving to Y: {target_pose['y']} with final yaw")
    #     self.go_to(
    #         target_x=target_pose['x'], 
    #         target_y=target_pose['y'], 
    #         yaw=target_pose['yaw']
    #     )

    def _wait_for_action_completion(self, action, ip=None):
        self._update_target(ip)
        while True:
            status = self.get_action_status(action["action_id"])
            # data_path.append(self.get_robot_pose())
            if not status or status["state"]["status"] == 4:
                return 0
    
    # def go_rectilinear_by_distance(self, angle, distance):
    #     # Ensure angle is one of the allowed rectilinear directions
    #     allowed_angles = [0, 90, 180, 270, 360]
    #     data_path = []
    #     if angle not in allowed_angles:
    #         print(f"Angle {angle} not in rectilinear range. Adjusting...")
    #         # Optional: round to nearest allowed angle
    #         angle = min(allowed_angles, key=lambda x: abs(x - angle)) % 360

    #     current_pose = self.get_robot_pose()
    #     original_yaw = current_pose['yaw']
    #     target_pose = compute_target_pose(angle, current_pose, distance)

    #     # --- LEG 2: MOVE Y ---
    #     # Re-check LIDAR before moving in the Y direction
    #     if self.is_obstacle_ahead(self.get_laserscan()):
    #         print("Obstacle detected! Performing U-Shape bypass before Y-move.")
    #         self.move_u_shape_around_obstacle(self.get_robot_pose(), data_path)

    #     print(f"Moving to Y: {target_pose['y']}")
    #     action = self.go_to(target_x=current_pose['x'], target_y=target_pose['y'], yaw=None)
    #     self._wait_for_action_completion(action=action, data_path=data_path)

    #     # --- LEG 1: MOVE X ---
    #     if self.is_obstacle_ahead(self.get_laserscan()):
    #         print("Obstacle detected! Performing U-Shape bypass before X-move.")
    #         self.move_u_shape_around_obstacle(self.get_robot_pose(), data_path)
    #         # Update current pose after bypass to recalculate X-leg
    #         current_pose = self.get_robot_pose()

    #     print(f"Moving to X: {target_pose['x']} with final yaw")
    #     action = self.go_to(target_x=target_pose['x'], target_y=target_pose['y'], yaw=original_yaw)
    #     self._wait_for_action_completion(action=action, data_path=data_path)

    #     self.plot_robot_path(data_list=data_path)

    # def is_obstacle_ahead(self, lidar_points, threshold=0.3):
    #     """Simple check for any points within 0.3m in the front 30-degree cone."""
    #     for pt in lidar_points:
    #         if pt['valid'] and pt['distance'] < threshold:
    #             # Check if angle is roughly straight ahead (approx -15 to +15 degrees)
    #             if abs(pt['angle']) < 0.26:
    #                 return True
    #     return False

    # def is_obstacle_to_the_left(self, lidar_points, threshold=0.6):
    #     """Check ~70 to 110 degrees (Left) to see if we passed the object."""
    #     for pt in lidar_points:
    #         if pt['valid'] and pt['distance'] < threshold:
    #             # 1.22 rad (70 deg) to 1.92 rad (110 deg)
    #             if 1.22 < pt['angle'] < 1.92:
    #                 return True
    #     return False

    # def move_u_shape_around_obstacle(self, current, step_size=0.2, safety_margin=0.3, data_path=[]):
    #     """
    #     Forces sharp corners by rotating to target yaw BEFORE moving linear distances.
    #     """
    #     start_x, start_y = current['x'], current['y']
    #     current_y = start_y

    #     # --- PHASE 1: DYNAMIC WIDTH (Strictly Vertical on Y) ---
    #     print("U-Shape: Rotating to 270° then side-stepping...")
    #     while True:
    #         if not self.is_obstacle_ahead(self.get_laserscan(), threshold=0.5):
    #             current_y -= step_size 
    #             # 1. Rotate in place first (by sending same X,Y but new Yaw)
    #             rotate_action = self.go_to(target_x=start_x, target_y=start_y, yaw=270)
    #             self._wait_for_action_completion(action=rotate_action, data_path=data_path)
    #             # 2. Move linear
    #             action = self.go_to(target_x=start_x, target_y=current_y, yaw=270)
    #             self._wait_for_action_completion(action=action, data_path=data_path)
    #             break
            
    #         current_y -= step_size 
    #         # Ensure we stay at start_x
    #         action = self.go_to(target_x=start_x, target_y=current_y, yaw=270)
    #         self._wait_for_action_completion(action=action, data_path=data_path)

    #     # --- PHASE 2: DYNAMIC DEPTH (Strictly Horizontal on X) ---
    #     print("U-Shape: Rotating to 0° then advancing...")
    #     # 1. Rotate in place to face forward before moving X
    #     rotate_action = self.go_to(target_x=start_x, target_y=current_y, yaw=0)
    #     self._wait_for_action_completion(action=rotate_action, data_path=data_path)
        
    #     current_x = start_x
    #     while True:
    #         current_x += step_size
    #         # Move linear while locked to the offset current_y
    #         action = self.go_to(target_x=current_x, target_y=current_y, yaw=0)
    #         self._wait_for_action_completion(action=action, data_path=data_path)
            
    #         if not self.is_obstacle_to_the_left(self.get_laserscan(), threshold=0.6):
    #             # Extra step to clear the tail
    #             current_x += step_size
    #             action = self.go_to(target_x=current_x, target_y=current_y, yaw=0)
    #             self._wait_for_action_completion(action=action, data_path=data_path)
    #             break

    #     # --- PHASE 3: RETURN (Strictly Vertical on Y) ---
    #     print("U-Shape: Rotating to 90° then returning to line...")
    #     # 1. Rotate in place to face the original track
    #     rotate_action = self.go_to(target_x=current_x, target_y=current_y, yaw=90)
    #     self._wait_for_action_completion(action=rotate_action, data_path=data_path)
    #     # 2. Move linear back to start_y
    #     action = self.go_to(target_x=current_x, target_y=start_y, yaw=90)
    #     self._wait_for_action_completion(action=action, data_path=data_path)

    #     # --- PHASE 4: FINAL RE-ALIGN ---
    #     # Rotate to face forward again before resuming main path
    #     final_rotate = self.go_to(target_x=current_x, target_y=start_y, yaw=0)
    #     self._wait_for_action_completion(action=final_rotate, data_path=data_path)

    # def go_rectilinear_by_distance(self, angle, distance):
    #     current_pose = self.get_robot_pose()
    #     target_pose = compute_target_pose(angle, current_pose, distance)

    #     # 1. Move to Target X (Maintain current Y)
    #     print(f"Moving to X: {target_pose['x']}")
    #     self.go_to(
    #         target_x=target_pose['x'], 
    #         target_y=current_pose['y'], 
    #         yaw=None # Don't worry about orientation yet
    #     )
        
    #     # 2. Move to Target Y (Reach final destination)
    #     print(f"Moving to Y: {target_pose['y']} with final yaw")
    #     self.go_to(
    #         target_x=target_pose['x'], 
    #         target_y=target_pose['y'], 
    #         yaw=target_pose['yaw']
    #     )
    # def plot_robot_path(self, data_list):
    #     # 1. Extract data directly from dictionaries
    #     if not data_list:
    #         print("Data list is empty. Nothing to plot.")
    #         return

    #     try:
    #         x_coords = [d['x'] for d in data_list]
    #         y_coords = [d['y'] for d in data_list]
    #         yaws = [d['yaw'] for d in data_list]
    #     except KeyError as e:
    #         print(f"Error: Dictionary missing key {e}")
    #         return

    #     plt.figure(figsize=(10, 7))
        
    #     # 2. Plot the continuous trajectory
    #     # Using a slightly thicker line and 'zorder' to keep it behind markers
    #     plt.plot(x_coords, y_coords, color='#1f77b4', label='Robot Path', linewidth=2, alpha=0.8, zorder=1)

    #     # 3. Add orientation arrows every 10 steps
    #     # This helps visualize which way the robot was facing during the U-shape bypass
    #     step = 10 
    #     for i in range(0, len(x_coords), step):
    #         dx = np.cos(yaws[i]) * 0.05
    #         dy = np.sin(yaws[i]) * 0.05
    #         plt.arrow(x_coords[i], y_coords[i], dx, dy, 
    #                 head_width=0.02, head_length=0.03, fc='black', ec='black', alpha=0.5)

    #     # 4. Highlight Start (Green) and End (Red) points
    #     plt.scatter(x_coords[0], y_coords[0], color='green', s=100, label='Start', zorder=5)
    #     plt.scatter(x_coords[-1], y_coords[-1], color='red', marker='X', s=100, label='End', zorder=5)

    #     # Formatting and Styling
    #     plt.xlabel("X Position (meters)")
    #     plt.ylabel("Y Position (meters)")
    #     plt.legend(loc='upper right')
    #     plt.grid(True, linestyle='--', alpha=0.6)
        
    #     # Ensures the path isn't stretched; 1m on X looks like 1m on Y
    #     plt.axis('equal') 
        
    #     # Save the plot
    #     output_file = "way_he_come.png"
    #     plt.savefig(output_file, dpi=300, bbox_inches='tight')
    #     plt.close()
    #     print(f"Plot successfully saved to {output_file}")


# class MSIS_PHOEBUS:
#     def __init__(self, ip="", port=80):
#         """
#         Initializes the AMR motion client.
#         """
#         self.port = port
#         self.ip = ip
#         self.api = PLAIN_API(ip, port)

#     def _update_target(self, ip):
#         """Helper to update API target IP if provided."""
#         if ip:
#             self.api.ip = ip # Assuming PLAIN_API allows IP updates

#     def shutdown_amr(self, ip=None):
#         self._update_target(ip)
#         payload = {"shutdown_time_interval": 0, "restart_time_interval": 0}
#         return self.api._post("/api/core/system/v1/power/:shutdown", payload)
    
#     def hibernate_amr(self, ip=None):
#         self._update_target(ip)
#         return self.api._post("/api/core/system/v1/power/:hibernate")
    
#     def wakeup_amr(self, ip=None):
#         self._update_target(ip)
#         return self.api._post("/api/core/system/v1/power/:wakeup")
    
#     def restart_amr(self, ip=None):
#         self._update_target(ip)
#         return self.api._post("/api/core/system/v1/power/:restartmodule")

#     def get_power_status(self, ip=None, _print=False):
#         self._update_target(ip)
#         response = self.api._get("/api/core/system/v1/power/status")
#         if _print: print(response)
#         return response

#     def get_robot_speed(self, ip=None, _print=False):
#         self._update_target(ip)
#         response = self.api._get("/api/core/motion/v1/speed")
#         if _print: print(response)
#         return response

#     def get_robot_pose(self, ip=None, _print=False):
#         self._update_target(ip)
#         response = self.api._get("/api/core/slam/v1/localization/pose")
#         if _print: print(response)
#         return response

#     def get_action_status(self, action_id, ip=None, _print=False):
#         self._update_target(ip)
#         response = self.api._get(f"/api/core/motion/v1/actions/{action_id}")
#         if _print: print(response)
#         return response

#     def stop_motion_now(self, ip=None):
#         self._update_target(ip)
#         return self.api._delete("/api/core/motion/v1/actions/:current")
    
#     def set_charge_station(self, ip=None):
#         self._update_target(ip)
#         payload = {"metadata": {"display_name": "string"}}
#         return self.api._post("/api/core/slam/v1/homedocks/:register", payload)
    
#     def go_to(self, target_x, target_y, yaw=None, ip=None, label="go_to"): 
#         self._update_target(ip)
#         move_options = {
#             "mode": 0,
#             "acceptable_precision": 0.0001,
#             "fail_retry_count": 0,
#             "flags": ["precise","with_yaw"],
#             "speed_ratio": 1.0,
#         }
#         if yaw is not None:
#             move_options["yaw"] = math.radians(yaw)

#         payload = {
#             "action_name": "slamtec.agent.actions.MoveToAction",
#             "options": {
#                 "target": {"x": target_x, "y": target_y, "z": 0},
#                 "move_options": move_options,
#             },
#         }
#         return self.api._post("/api/core/motion/v1/actions", payload)

#     def go_by_distance(self, angle, distance, ip=None, label="go_by_distance"):
#         self._update_target(ip)
#         current_pose = self.get_robot_pose(ip=ip)
#         target_pose = compute_target_pose(angle, current_pose, distance)

#         payload = {
#             "action_name": "slamtec.agent.actions.MoveToAction",
#             "options": {
#                 "target": {"x": target_pose['x'], "y": target_pose['y'], "z": 0},
#                 "move_options": {
#                     "mode": 0,
#                     "acceptable_precision": 0.1,
#                     "fail_retry_count": 0,
#                     "speed_ratio": 1.0,
#                 },
#             },
#         }
#         return self.api._post("/api/core/motion/v1/actions", payload)
    
#     def rotate_by_inPlace(self, deg=0, ip=None):
#         self._update_target(ip)
#         payload = {
#             "action_name": "slamtec.agent.actions.RotateAction", 
#             "options": {"angle": math.radians(deg)}
#         }
#         return self.api._post("/api/core/motion/v1/actions", payload)

#     def rotate_to_inPlace(self, deg=0, unit='deg', ip=None):
#         self._update_target(ip)
#         angle = math.radians(deg) if unit=='deg' else deg
#         payload = {
#             "action_name": "slamtec.agent.actions.RotateToAction", 
#             "options": {"angle": angle}
#         }
#         return self.api._post("/api/core/motion/v1/actions", payload)

#     def go_charge(self, ip=None):
#         self._update_target(ip)
#         payload = {
#             "action_name": "slamtec.agent.actions.GoHomeAction",
#             "options": {
#                 "flags": "dock",
#                 "back_to_landing": True,
#                 "charging_retry_count": 2,
#                 "move_options": {"mode": 0},
#             },
#         }
#         return self.api._post("/api/core/motion/v1/actions", payload)

#     def get_laserscan(self, ip=None):
#         self._update_target(ip)
#         response = self.api._get("/api/core/system/v1/laserscan")
#         return response.get("laser_points", [])
    
#     def _wait_for_action_completion(self, action):
#         while True:
#             status = self.get_action_status(action["action_id"])
#             if not status or status["state"]["status"] == 4:
#                 return 0
    
if __name__ == "__main__":
    amr = MSIS_PHOEBUS(ip="192.168.0.132", port=1448)

    # 1. Initialize the figure and axis once
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_title("Real-time Lidar Scan", va='bottom')

    # Create empty plot objects that we will update later
    scatter = ax.scatter([], [], color='red', s=10, label='Obstacles')
    line, = ax.plot([], [], color='blue', linestyle='--', alpha=0.5)

    ax.grid(True)
    ax.legend(loc='upper right')

    # Turn on interactive mode
    plt.ion()
    plt.show()

    try:
        while True:
            lidar_points = amr.get_laserscan()
            
            if not lidar_points:
                continue

            # Extract data
            angles = [d['angle'] for d in lidar_points]
            distances = [d['distance'] for d in lidar_points]

            # 2. Update the data of the existing objects
            # Update scatter plot: offsets must be a (N, 2) array of (angle, distance)
            offsets = np.c_[angles, distances]
            scatter.set_offsets(offsets)
            
            # Update the connecting line
            line.set_data(angles, distances)

            # 3. Dynamically adjust the radial limit if necessary
            max_dist = max(distances) if distances else 1
            ax.set_ylim(0, max_dist * 1.1)

            # 4. Use draw and flush_events instead of plt.show()
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            # Optional: Add a tiny pause to allow the GUI to process
            plt.pause(0.01)

    except KeyboardInterrupt:
        print("Stopping visualization...")
        plt.ioff()
        plt.show()