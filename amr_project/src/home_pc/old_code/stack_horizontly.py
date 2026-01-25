import time
import threading
import json, yaml
import copy
import math
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from fairino import Robot

from msis_amr_motion.msis_amr_motion import MSIS_PHOEBUS
from msis_amr_motion.util import compute_distance_and_angle

"""
MSIS AMR TEAM Project

Features:
- Control buttons: STOP, PAUSE, RESUME
- 6 joint movement buttons (MoveJ for each joint)
- Save/load pick and place poses (Cartesian coordinates)
- Configurable number of boxes and box height
- Pick-and-place stacking automation
"""

# ===== Global robot instance and flags =====
robot = None
stop_flag = False
pause_flag = False
robot_lock = threading.Lock()  #to prevent simultanous concurrent robot commands

# ===== Default configuration =====
DEFAULT_ROBOT_IP = "192.168.0.15"
GRIPPER_INDEX = 1
GRIP_SPEED = 90
GRIP_FORCE = 50
GRIP_MAX_TIME = 30000
GRIP_BLOCK = 0

TOOL = 0
USER = 0
VEL = 90
VEL_SLOW = 30
BLENDT = 100

# Joint increments (degrees)
JOINT_INCREMENT = 5 #change based on how much u want to move joint on a single click

# Default poses (Cartesian: [x, y, z, rx, ry, rz])
home_pose= [89.690, 116.508, 846.874, -177.624, 52.775, -79.285]#[182.082, 340.817, 475.358, 175.088, 2.385, 142.770]#check to avoid crash
#[49.385, -123.93, 35.491, -56.425, -92.664, 91.885]
cam_pose=[391.242, -111.323, 749.343, 179.968, 1.152, -179.211]
thaising_pose=[9.132, -420.182, 713.185, -179.729, -0.313, 94.811]  #for detecting number of boxes on amr body
DEFAULT_NUM_BOXES = 4
DEFAULT_BOX_HEIGHT = 58.0
DEFAULT_APPROACH_OFFSET = 40.0
DEFAULT_COLUMN_OFFSET = 100.0
Z_THRESHHOLD = 321  #
POSES_FILE = "saved_poses_horizontal_stack.json"
PUSH='LEFT'
############ AMR #################
with open("D:/Robot/fairino-python-sdk-main/fairino-python-sdk-main/windows/msis_amr_motion/config.yaml", 'r') as stream:
    try:
        # Use safe_load for security when loading from untrusted sources
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
AMR = MSIS_PHOEBUS(ip=config['amr']['Banana'], port=config['amr']['port'])
##################################


# ===== Robot connection =====
def connect_robot(ip):
    global robot
    try:
        robot = Robot.RPC(ip)
        robot.SetSpeed(VEL)
        robot.ActGripper(GRIPPER_INDEX, 1)
        return True, "Robot connected successfully"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"


def disconnect_robot():
    global robot
    if robot:
        try:
            robot.CloseRPC()
            robot = None
            return True, "Robot disconnected"
        except Exception as e:
            return False, f"Disconnect failed: {str(e)}"
    return False, "Robot not connected"


# ===== Safety controls =====
def stop_robot():
    """Stop robot immediately - need to be fixed."""
    global stop_flag, pause_flag
    stop_flag = True
    pause_flag = False
    if robot:
        try:
            robot.StopMotion()
        except Exception:
            pass


def pause_robot():
    """Pause robot immediately - ."""
    global pause_flag
    pause_flag = True
    if robot:
        try:
            robot.PauseMotion()
        except Exception:
            pass


def resume_robot():
    """Resume paused robot immediately - does NOT hold lock."""
    global pause_flag
    pause_flag = False
    if robot:
        try:
            robot.ResumeMotion()
        except Exception:
            pass

def go_home(pose):#if called should go to the predifined home pose
    if stop_flag:
        return False
    if not move_cart(pose):
        return False 


def reset_flags():
    global stop_flag, pause_flag
    with robot_lock:
        stop_flag = False
        pause_flag = False


def wait_while_paused():
    """Wait while robot is paused or stopped. Does NOT acquire lock."""
    while True:
        if stop_flag:
            return False
        if not pause_flag:
            return True
        time.sleep(0.05)


# ===== Gripper controls =====
def open_gripper():
    if not robot:
        return False
    if stop_flag or pause_flag:
        return False
    try:
        with robot_lock:
            robot.MoveGripper(GRIPPER_INDEX, 100, GRIP_SPEED, GRIP_FORCE, GRIP_MAX_TIME, GRIP_BLOCK, 0, 0, 0, 0)
        return True
    except Exception as e:
        print(f"Open gripper error: {e}")
        return False


def close_gripper(angle=23):#cuurently 60 for white box     
    if not robot:
        return False
    if stop_flag or pause_flag:
        return False
    try:
        with robot_lock:
            robot.MoveGripper(GRIPPER_INDEX, angle, GRIP_SPEED, GRIP_FORCE, GRIP_MAX_TIME, GRIP_BLOCK, 0, 0, 0, 0)
        return True
    except Exception as e:
        print(f"Close gripper error: {e}")
        return False


# ===== Joint movement =====
def move_joint_worker(joint_index, direction, gui_callback=None):
    """Worker function: move a specific joint (runs in background thread).
    joint_index: 0-5 (joint 1-6)
    direction: 1 for positive, -1 for negative
    gui_callback: function to call with (title, message, is_error)
    """
    if not robot:
        if gui_callback:
            gui_callback("Error", "Robot not connected", True)
        return

    try:
        with robot_lock:
            # Get current joint state
            ret, jpos = robot.GetActualJointPosDegree()
            if ret == 0 and jpos:
                # Modify target joint
                target_jpos = list(jpos)
                target_jpos[joint_index] += direction * JOINT_INCREMENT

                # MoveJ with joint coordinates
                robot.MoveJ(joint_pos=target_jpos, tool=TOOL, user=USER, vel=VEL, blendT=BLENDT)
        
        # Wait for motion to complete (outside lock)
        if not wait_while_paused():
            if gui_callback:
                gui_callback("Info", "Joint movement stopped", False)
            return

        if gui_callback:
            gui_callback("Success", f"Joint {joint_index + 1} moved {direction * JOINT_INCREMENT}°", False)
    except Exception as e:
        if gui_callback:
            gui_callback("Error", f"Joint move failed: {str(e)}", True)


def move_joint(joint_index, direction, gui_callback=None):
    """Trigger joint movement in background thread to prevent GUI freeze."""
    if not robot:
        messagebox.showerror("Error", "Robot not connected")
        return
    
    # Run in background thread
    thread = threading.Thread(target=move_joint_worker, args=(joint_index, direction, gui_callback), daemon=True)
    thread.start()


# ===== Cartesian movement =====
def move_cart(pose, velocity=None):
    """Move to Cartesian pose and wait. velocity parameter overrides default VEL."""
    if not robot:
        return False
    
    if stop_flag:
        return False

    try:
        vel = velocity if velocity is not None else VEL
        with robot_lock:
            robot.MoveCart(desc_pos=pose, tool=TOOL, user=USER, vel=vel, blendT=BLENDT)

        # Wait while paused or for stop (outside lock)
        if not wait_while_paused():
            return False

        return True
    except Exception as e:
        print(f"Cartesian move error: {e}")
        return False


def get_current_pose():
    """Get current Cartesian pose."""
    if not robot:
        return None
    try:
        with robot_lock:
            ret, pose = robot.GetActualTCPPose()
            if ret == 0 and pose:
                return pose
    except Exception as e:
        print(f"Get pose error: {e}")
    return None


# ===== Pose save/load =====
def load_poses():
    """Load saved poses from file."""
    try:
        with open(POSES_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = None
    except Exception as e:
        print(f"Load error: {e}")
        data = None

    # Default structure with 5 tables
    def default_tables():
        return {}

    if not data:
        return {
            "tables": default_tables(),
            "num_boxes": DEFAULT_NUM_BOXES,
            "box_height": DEFAULT_BOX_HEIGHT,
            "approach_offset": DEFAULT_APPROACH_OFFSET,
            "column_offset": DEFAULT_COLUMN_OFFSET
        }

    # Backwards compatibility: old flat format -> migrate into table '1'
    if any(k in data for k in ("pick_pose", "place_pose", "safe_pick", "safe_place")):
        tables = default_tables()
        tables["1"] = {
            "pick_pose": data.get("pick_pose"),
            "place_pose": data.get("place_pose"),
            "safe_pick": data.get("safe_pick"),
            "safe_place": data.get("safe_place")
        }
        return {
            "tables": tables,
            "num_boxes": data.get("num_boxes", DEFAULT_NUM_BOXES),
            "box_height": data.get("box_height", DEFAULT_BOX_HEIGHT),
            "approach_offset": data.get("approach_offset", DEFAULT_APPROACH_OFFSET),
            "column_offset": data.get("column_offset", DEFAULT_COLUMN_OFFSET)
        }

    # If data already in new format, ensure tables exist
    if "tables" not in data:
        data["tables"] = default_tables()
    else:
        # No filling missing tables with defaults
        pass

    # Ensure top-level params
    data.setdefault("num_boxes", DEFAULT_NUM_BOXES)
    data.setdefault("box_height", DEFAULT_BOX_HEIGHT)
    data.setdefault("approach_offset", DEFAULT_APPROACH_OFFSET)
    data.setdefault("column_offset", DEFAULT_COLUMN_OFFSET)
    data.setdefault("vel", VEL)
    data.setdefault("vel_slow", VEL_SLOW)

    return data


def save_poses(data):
    """Save poses to file."""
    try:
        with open(POSES_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"Save error: {e}")
        return False


# ===== Pick and place logic =====
def set_pose_z(pose, z):
    p = copy.deepcopy(pose)
    p[2] = z
    return p
def set_pose_y(pose, y):
    p = copy.deepcopy(pose)
    p[1] = y
    return p
def set_pose_x(pose, x):
    p = copy.deepcopy(pose)
    p[0] = x
    return p
def set_pose_y_z(pose, y, z):
    p = copy.deepcopy(pose)
    p[1] = y
    p[2] = z
    return p

def pick_and_place_one(pick_base, place_base, safe_pick, safe_place, pick_z, place_z, place_y, push_start, push_end, approach_offset, vel_normal=None, vel_slow=None, status_callback=None):
    """Performs pick and place for a single box using safe waypoint poses.
    vel_normal: normal movement speed (default VEL)
    vel_slow: slow movement speed for fine approach (default VEL)
    """
    if stop_flag:
        return False

    vel_n = vel_normal if vel_normal is not None else VEL
    vel_s = vel_slow if vel_slow is not None else VEL

    target_pick = set_pose_z(pick_base, pick_z)
    target_place = set_pose_y_z(place_base, place_y, place_z)
    push_s = set_pose_y_z(place_base, push_start, place_z)
    push_e = set_pose_y_z(place_base, push_end, place_z)
   # place_z = target_place[2]
    approach_pick = set_pose_z(target_pick, pick_z + approach_offset)
    approach_place = set_pose_z(target_place, place_z + approach_offset)
    # Move to safe pick position first
    if status_callback:
        status_callback("Moving to safe pick position...")
    if not move_cart(safe_pick, vel_n):
        return False
    if status_callback:
        status_callback("Moving to safe pick position...")
    if not move_cart(approach_pick, vel_n):
        return False
    # Approach and pick (use slow speed for final approach)
    if status_callback:
        status_callback("Approaching pick position...")
    if not move_cart(target_pick, vel_s):
        return False
    
    close_gripper()
    time.sleep(0.15)

    if status_callback:
        status_callback("Moving to safe pick position...")
    if not move_cart(approach_pick, vel_n):
        return False
    if status_callback:
        status_callback("Closing gripper...")


    # Return to safe pick position
    if status_callback:
        status_callback("Returning to safe position...")
    if not move_cart(safe_pick, vel_n):
        return False

    # Move to safe place position
    if status_callback:
        status_callback("Moving to safe place position...")
    if not move_cart(safe_place, vel_n):
        return False

    # Approach and place (use slow speed for final approach)
    if status_callback:
        status_callback("Approaching  place ...")
    if not move_cart(approach_place, vel_n):
        return False


    if status_callback:
        status_callback("Approaching place position...")
    if not move_cart(target_place, vel_s):
        return False
    if status_callback:
        status_callback("Opening gripper...")
    open_gripper()
    time.sleep(0.15)

    if status_callback:
        status_callback("Approaching  place after place ...")
    if not move_cart(approach_place, vel_n):
        return False
    #push start
    if status_callback:
        status_callback("Pushing box to align...")
    if not move_cart(push_s, vel_s):
        return False
    #push end
    if status_callback:
        status_callback("Pushing box to align end...")  
    if not move_cart(push_e, 10):
        return False
    if status_callback: 
        status_callback("Approaching  place after place ...")
    if not move_cart(approach_place, vel_s):
        return False
    # Return to safe place position
    if status_callback:
        status_callback("Returning to safe position...")
    if not move_cart(safe_place, vel_n):
        return False

    return True


def column_stacking_sequence(pick_pose, place_pose, safe_pick, safe_place, num_boxes, box_height, approach_offset, status_callback=None, vel_normal=None, vel_slow=None):
    """Run the stacking sequence in background -."""
    global stop_flag, pause_flag

    reset_flags()
  
    for i in range(num_boxes):
        # Check for stop before each box
        if stop_flag:
            if status_callback:
                status_callback("Operation stopped by user")
            break

        # Wait if paused
        while pause_flag and not stop_flag:
            if status_callback:
                status_callback("Paused - waiting for resume...")
            time.sleep(0.2)
        
        pick_z = pick_pose[2] - i * box_height
        if PUSH=='LEFT':
            place_y = place_pose[1] - i * 80
            push_start=place_y-150
            push_end =place_y+100
        else:
            place_y = place_pose[1] + i * 80
            push_start=place_y+150
            push_end =place_y-100

        if status_callback:
            status_callback(f"\n--- Box {i+1}/{num_boxes} ---")
            status_callback(f"Pick Z: {pick_z:.3f}  Place Y: {place_y:.3f}")
        ok = pick_and_place_one(pick_pose, place_pose, safe_pick, safe_place, pick_z, place_y, push_start, push_end, approach_offset, vel_normal, vel_slow, status_callback)
        if not ok:
            if status_callback:
                status_callback("Pick-and-place aborted")
            break

        time.sleep(0.2)

    if status_callback:
        status_callback("Stacking sequence completed")

#def column_stacking_sequence():  
def run_stacking_sequence(pick_pose, place_pose, safe_pick, safe_place, num_boxes, box_height, approach_offset, status_callback=None, vel_normal=None, vel_slow=None, column_offset=None):
    """Run the stacking sequence in background -."""
    global stop_flag, pause_flag

    if column_offset is None:
        column_offset = DEFAULT_COLUMN_OFFSET

    reset_flags()
    for j in range(2):
        pick_x_offset = -j * column_offset
        pick_pose_offset = set_pose_x(pick_pose, pick_pose[0] + pick_x_offset)
        safe_pick_offset = set_pose_x(safe_pick, safe_pick[0] + pick_x_offset)
        for i in range(num_boxes-j):
            
            if stop_flag:
                if status_callback:
                    status_callback("Operation stopped by user")
                break

            
            while pause_flag and not stop_flag:
                if status_callback:
                    status_callback("Paused - waiting for resume...")
                time.sleep(0.2)
            
            pick_z = pick_pose_offset[2] - i * box_height
            place_z = place_pose[2] + j * box_height
            if PUSH=='LEFT':
                place_y = place_pose[1] - i * 74 +j*74 #+j*40
                push_start=place_y-150
                push_end =(place_y+37)-(j*74) 
            else:
                place_y = place_pose[1] + i * 80
                push_start=place_y+150
                push_end =place_y-100

            if status_callback: 
                status_callback(f"\n--- Box {i+1}/{num_boxes} ---") 
                status_callback(f"Pick Z: {pick_z:.3f}  Place Y: {place_y:.3f}")
            ok = pick_and_place_one(pick_pose_offset, place_pose, safe_pick_offset, safe_place, pick_z, place_z, place_y, push_start, push_end, approach_offset, vel_normal, vel_slow, status_callback)
            if not ok:
                if status_callback:
                    status_callback("Pick-and-place aborted")
                break

            time.sleep(0.2)

        if status_callback:
            status_callback("Stacking sequence completed")
# ===== GUI  =====
class RobotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MSIS Robot Control")
        self.root.geometry("900x800")

        self.poses_data = load_poses()

        
        self.left_frame = ttk.Frame(root)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ===== Connection Frame =====
        self.conn_frame = ttk.LabelFrame(self.left_frame, text="Robot Connection", padding=10)
        self.conn_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(self.conn_frame, text="Robot IP:").grid(row=0, column=0, sticky=tk.W)
        self.ip_var = tk.StringVar(value=DEFAULT_ROBOT_IP)
        ttk.Entry(self.conn_frame, textvariable=self.ip_var, width=20).grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Button(self.conn_frame, text="Connect", command=self.connect).grid(row=0, column=2, padx=5)
        ttk.Button(self.conn_frame, text="Disconnect", command=self.disconnect).grid(row=0, column=3, padx=5)

        self.status_label = ttk.Label(self.conn_frame, text="Status: Disconnected", foreground="red")
        self.status_label.grid(row=0, column=4, padx=10)

        # ===== Control Frame =====
        self.ctrl_frame = ttk.LabelFrame(self.left_frame, text="Robot Controls", padding=10)
        self.ctrl_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(self.ctrl_frame, text="STOP", command=stop_robot, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.ctrl_frame, text="PAUSE", command=pause_robot, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.ctrl_frame, text="RESUME", command=resume_robot, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.ctrl_frame, text="GO HOME", command=lambda: self.go_home_action(), width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.ctrl_frame, text="AMR table1", command=lambda: self.AMR_go_to(point_name="Table1"), width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.ctrl_frame, text="AMR table2", command=lambda: self.AMR_go_to(point_name="Table2"), width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.ctrl_frame, text="AMR stop", command=lambda: AMR.stop_motion_now(), width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.ctrl_frame, text="CAM1", command=lambda: self.go_cam1_action(), width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.ctrl_frame, text="CAM2", command=lambda: self.go_cam2_action(), width=10).pack(side=tk.LEFT, padx=5)
# ######
#         ttk.Label(self.ctrl_frame, text="Table:").pack(side=tk.LEFT, padx=5)
#         self.table_var = tk.StringVar(value="1")
#         self.table_combo = ttk.Combobox(self.ctrl_frame, textvariable=self.table_var, values=["1","2"], width=5, state="readonly")
#         self.table_combo.grid(row=0, column=7, sticky=tk.W, padx=5)

#####

        # ===== Joint Movement Frame =====
        self.joint_frame = ttk.LabelFrame(self.left_frame, text="Joint Movement (±5°)", padding=10)
        self.joint_frame.pack(fill=tk.X, padx=10, pady=5)

        for i in range(6):
            sub_frame = ttk.Frame(self.joint_frame)
            sub_frame.pack(fill=tk.X, padx=5, pady=3)

            ttk.Label(sub_frame, text=f"Joint {i+1}:", width=10).pack(side=tk.LEFT)
            ttk.Button(sub_frame, text="←", width=3, command=lambda j=i: move_joint(j, -1, self.joint_callback)).pack(side=tk.LEFT, padx=2)
            ttk.Button(sub_frame, text="→", width=3, command=lambda j=i: move_joint(j, 1, self.joint_callback)).pack(side=tk.LEFT, padx=2)

        # ===== Gripper Frame =====
        self.gripper_frame = ttk.LabelFrame(self.left_frame, text="Gripper Control", padding=10)
        self.gripper_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(self.gripper_frame, text="Open Gripper", command=lambda: open_gripper(), width=20).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.gripper_frame, text="Close Gripper", command=lambda: close_gripper(), width=20).pack(side=tk.LEFT, padx=5)

        # ===== Current Pose Frame =====
        self.pose_frame = ttk.LabelFrame(self.left_frame, text="Current Pose (Cartesian)", padding=10)
        self.pose_frame.pack(fill=tk.X, padx=10, pady=5)

        self.pose_label = ttk.Label(self.pose_frame, text="Not available", font=("Courier", 9))
        self.pose_label.pack()

        ttk.Button(self.pose_frame, text="Refresh Pose", command=self.refresh_pose).pack(pady=5)

        # ===== Pose Save/Load Frame =====
        self.save_frame = ttk.LabelFrame(self.left_frame, text="Pose Management", padding=10)
        self.save_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(self.save_frame, text="Save Pick Pose", command=self.save_pick_pose, width=18).pack(side=tk.LEFT, padx=3)
        ttk.Button(self.save_frame, text="Save Safe Pick", command=self.save_safe_pick_pose, width=18).pack(side=tk.LEFT, padx=3)
        ttk.Button(self.save_frame, text="Save Place Pose", command=self.save_place_pose, width=18).pack(side=tk.LEFT, padx=3)
        ttk.Button(self.save_frame, text="Save Safe Place", command=self.save_safe_place_pose, width=18).pack(side=tk.LEFT, padx=3)
        ttk.Button(self.save_frame, text="View Saved Poses", command=self.view_poses, width=18).pack(side=tk.LEFT, padx=3)

        # ===== Stacking Parameters Frame =====
        self.param_frame = ttk.LabelFrame(self.left_frame, text="Stacking Parameters", padding=10)
        self.param_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(self.param_frame, text="Number of Boxes:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.num_boxes_var = tk.IntVar(value=self.poses_data.get("num_boxes", DEFAULT_NUM_BOXES))
        ttk.Entry(self.param_frame, textvariable=self.num_boxes_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5)

        ttk.Label(self.param_frame, text="Box Height (mm):").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.box_height_var = tk.DoubleVar(value=self.poses_data.get("box_height", DEFAULT_BOX_HEIGHT))
        ttk.Entry(self.param_frame, textvariable=self.box_height_var, width=10).grid(row=0, column=3, sticky=tk.W, padx=5)

        ttk.Label(self.param_frame, text="Approach Offset (mm):").grid(row=0, column=4, sticky=tk.W, padx=5)
        self.offset_var = tk.DoubleVar(value=self.poses_data.get("approach_offset", DEFAULT_APPROACH_OFFSET))
        ttk.Entry(self.param_frame, textvariable=self.offset_var, width=10).grid(row=0, column=5, sticky=tk.W, padx=5)

        ttk.Label(self.param_frame, text="Column Offset (mm):").grid(row=1, column=5, sticky=tk.W, padx=5)
        self.column_offset_var = tk.DoubleVar(value=self.poses_data.get("column_offset", DEFAULT_COLUMN_OFFSET))
        ttk.Entry(self.param_frame, textvariable=self.column_offset_var, width=10).grid(row=1, column=6, sticky=tk.W, padx=5)

        # Table selector (1-5)
        ttk.Label(self.param_frame, text="Table:").grid(row=0, column=8, sticky=tk.W, padx=5)
        self.table_var = tk.StringVar(value="1")
        self.table_combo = ttk.Combobox(self.param_frame, textvariable=self.table_var, values=["1","2","3","4","5"], width=5, state="readonly")
        self.table_combo.grid(row=0, column=9, sticky=tk.W, padx=5)

        # Speed controls -  second row
        ttk.Label(self.param_frame, text="Normal Speed (vel):").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.vel_var = tk.DoubleVar(value=self.poses_data.get("vel", VEL))
        ttk.Entry(self.param_frame, textvariable=self.vel_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5)

        ttk.Label(self.param_frame, text="Slow Speed (vel_slow):").grid(row=1, column=2, sticky=tk.W, padx=5)
        self.vel_slow_var = tk.DoubleVar(value=self.poses_data.get("vel_slow", VEL))
        ttk.Entry(self.param_frame, textvariable=self.vel_slow_var, width=10).grid(row=1, column=3, sticky=tk.W, padx=5)

        ttk.Button(self.param_frame, text="Save Parameters", command=self.save_parameters).grid(row=1, column=4, columnspan=2, sticky=tk.W, padx=5)

        # ===== Stacking Control Frame =====
        self.run_frame = ttk.LabelFrame(self.left_frame, text="Stacking Control", padding=10)
        self.run_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(self.run_frame, text="Start Stacking", command=self.start_stacking, width=20).pack(side=tk.LEFT, padx=5)

        # ===== Status Output Frame =====
        
        self.output_frame = ttk.LabelFrame(root, text="Status Output", padding=10)
        self.output_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.output_text = tk.Text(self.output_frame, height=20, width=60, font=("Courier", 9))
        self.output_text.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(self.output_frame, orient=tk.VERTICAL, command=self.output_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.config(yscrollcommand=scrollbar.set)

    def connect(self):
        ip = self.ip_var.get()
        ok, msg = connect_robot(ip)
        if ok:
            self.status_label.config(text="Status: Connected", foreground="green")
            self.log(f"✓ {msg}")
        else:
            self.status_label.config(text="Status: Failed", foreground="red")
            self.log(f"✗ {msg}")
            messagebox.showerror("Connection Error", msg)

    def joint_callback(self, title, message, is_error):
        """Callback for joint movement status (called from worker thread)."""
        if is_error:
            self.log(f"✗ {message}")
        else:
            self.log(f"✓ {message}")

    def disconnect(self):
        ok, msg = disconnect_robot()
        if ok:
            self.status_label.config(text="Status: Disconnected", foreground="red")
            self.log(f"✓ {msg}")
        else:
            self.log(f"✗ {msg}")

    def refresh_pose(self):
        pose = get_current_pose()
        if pose:
            pose_str = f"[{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}, {pose[3]:.3f}, {pose[4]:.3f}, {pose[5]:.3f}]"
            self.pose_label.config(text=pose_str)
            self.log(f"Current pose: {pose_str}")
        else:
            self.log("Could not retrieve current pose")
            messagebox.showerror("Error", "Could not get current pose")

    def save_pick_pose(self):
        pose = get_current_pose()
        if pose:
            table = self.table_var.get()
            self.poses_data.setdefault("tables", {})
            self.poses_data["tables"].setdefault(table, {})
            self.poses_data["tables"][table]["pick_pose"] = pose
            if save_poses(self.poses_data):
                self.log(f"✓ Pick pose saved (Table {table}): {pose}")
                messagebox.showinfo("Success", f"Pick pose saved for Table {table}")
            else:
                self.log("✗ Failed to save pick pose")
                messagebox.showerror("Error", "Failed to save pose")
        else:
            messagebox.showerror("Error", "Could not get current pose")

    def save_safe_pick_pose(self):
        pose = get_current_pose()
        if pose:
            table = self.table_var.get()
            self.poses_data.setdefault("tables", {})
            self.poses_data["tables"].setdefault(table, {})
            self.poses_data["tables"][table]["safe_pick"] = pose
            if save_poses(self.poses_data):
                self.log(f"✓ Safe pick pose saved (Table {table}): {pose}")
                messagebox.showinfo("Success", f"Safe pick pose saved for Table {table}")
            else:
                self.log("✗ Failed to save safe pick pose")
                messagebox.showerror("Error", "Failed to save pose")
        else:
            messagebox.showerror("Error", "Could not get current pose")

    def save_place_pose(self):
        pose = get_current_pose()
        if pose:
            table = self.table_var.get()
            self.poses_data.setdefault("tables", {})
            self.poses_data["tables"].setdefault(table, {})
            self.poses_data["tables"][table]["place_pose"] = pose
            if save_poses(self.poses_data):
                self.log(f"✓ Place pose saved (Table {table}): {pose}")
                messagebox.showinfo("Success", f"Place pose saved for Table {table}")
            else:
                self.log("✗ Failed to save place pose")
                messagebox.showerror("Error", "Failed to save pose")
        else:
            messagebox.showerror("Error", "Could not get current pose")

    def save_safe_place_pose(self):
        pose = get_current_pose()
        if pose:
            table = self.table_var.get()
            self.poses_data.setdefault("tables", {})
            self.poses_data["tables"].setdefault(table, {})
            self.poses_data["tables"][table]["safe_place"] = pose
            if save_poses(self.poses_data):
                self.log(f"✓ Safe place pose saved (Table {table}): {pose}")
                messagebox.showinfo("Success", f"Safe place pose saved for Table {table}")
            else:
                self.log("✗ Failed to save safe place pose")
                messagebox.showerror("Error", "Failed to save pose")
        else:
            messagebox.showerror("Error", "Could not get current pose")

    def view_poses(self):
        self.log("=== Saved Poses ===")
        table = self.table_var.get() if hasattr(self, 'table_var') else '1'
        tables = self.poses_data.get('tables', {})
        if table in tables:
            t = tables[table]
            self.log(f"Table {table} - Pick: {t.get('pick_pose')}")
            self.log(f"Table {table} - Safe Pick: {t.get('safe_pick')}")
            self.log(f"Table {table} - Place: {t.get('place_pose')}")
            self.log(f"Table {table} - Safe Place: {t.get('safe_place')}")
        else:
            self.log(f"Table {table} not configured")

        self.log(f"Num Boxes: {self.poses_data.get('num_boxes')}")
        self.log(f"Box Height: {self.poses_data.get('box_height')}")
        self.log(f"Approach Offset: {self.poses_data.get('approach_offset')}")
        self.log(f"Column Offset: {self.poses_data.get('column_offset')}")
        self.log(f"Normal Speed (vel): {self.poses_data.get('vel')}")
        self.log(f"Slow Speed (vel_slow): {self.poses_data.get('vel_slow')}")

    def save_parameters(self):
        """Save all stacking parameters to poses file."""
        try:
            self.poses_data["num_boxes"] = self.num_boxes_var.get()
            self.poses_data["box_height"] = self.box_height_var.get()
            self.poses_data["approach_offset"] = self.offset_var.get()
            self.poses_data["column_offset"] = self.column_offset_var.get()
            self.poses_data["vel"] = self.vel_var.get()
            self.poses_data["vel_slow"] = self.vel_slow_var.get()
            
            if save_poses(self.poses_data):
                self.log("✓ Parameters saved successfully")
                messagebox.showinfo("Success", "Parameters saved!")
            else:
                self.log("✗ Failed to save parameters")
                messagebox.showerror("Error", "Failed to save parameters")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid parameters: {str(e)}")

    def start_stacking(self):
        if not robot:
            messagebox.showerror("Error", "Robot not connected")
            return

        try:
            num_boxes = self.num_boxes_var.get()
            box_height = self.box_height_var.get()
            approach_offset = self.offset_var.get()
            column_offset = self.column_offset_var.get()
            vel_normal = self.vel_var.get()
            vel_slow = self.vel_slow_var.get()

            # select poses from currently selected table
            table = self.table_var.get() if hasattr(self, 'table_var') else '1'
            tables = self.poses_data.get('tables', {})
            tbl = tables.get(table, {})
            pick_pose = tbl.get('pick_pose')
            place_pose = tbl.get('place_pose')
            safe_pick = tbl.get('safe_pick')
            safe_place = tbl.get('safe_place')

            if not pick_pose or not place_pose or not safe_pick or not safe_place:
                messagebox.showerror("Error", f"Missing poses for table {table}. Please save all poses first.")
                return

            self.log(f"Starting stacking: {num_boxes} boxes, height {box_height} mm")
            self.log(f"Speed: normal={vel_normal}, slow={vel_slow}")

            # Run in background thread
            def stacking_thread():
                run_stacking_sequence(pick_pose, place_pose, safe_pick, safe_place, num_boxes, box_height, approach_offset, self.log, vel_normal, vel_slow, column_offset)

            thread = threading.Thread(target=stacking_thread, daemon=True)
            thread.start()

        except Exception as e:
            messagebox.showerror("Error", f"Invalid parameters: {str(e)}")
    def go_cam1_action(self):
        """Move robot to home pose in background thread."""
        if not robot:
            messagebox.showerror("Error", "Robot not connected")
            return
        
        def home_thread():
            if go_home(cam_pose):
                self.log("✓ Moved to home position")
            else:
                self.log("✗ Failed to move to home position (stopped or paused)")
        
        thread = threading.Thread(target=home_thread, daemon=True)
        thread.start()
    def go_cam2_action(self):
        """Move robot to home pose in background thread."""
        if not robot:
            messagebox.showerror("Error", "Robot not connected")
            return
        
        def home_thread():
            if go_home(thaising_pose):
                self.log("✓ Moved to home position")
            else:
                self.log("✗ Failed to move to home position (stopped or paused)")
        
        thread = threading.Thread(target=home_thread, daemon=True)
        thread.start()

    def go_home_action(self):
        """Move robot to home pose in background thread."""
        if not robot:
            messagebox.showerror("Error", "Robot not connected")
            return
        
        def home_thread():
            if go_home(home_pose):
                self.log("✓ Moved to home position")
            else:
                self.log("✗ Failed to move to home position (stopped or paused)")
        
        thread = threading.Thread(target=home_thread, daemon=True)
        thread.start()

    def log(self, message):
        self.output_text.insert(tk.END, f"{message}\n")
        self.output_text.see(tk.END)
        self.root.update()

    def AMR_go_to(self, point_name):
        AMR.set_speed_limit(5.0)

        if point_name == "Table1":
            action = AMR.go_to(target_x=config['points']['Rotate1']['x'], target_y=config['points']['Rotate1']['y'], yaw=config['points']['Rotate1']['yaw'], speed_ratio=2.0)
            AMR._wait_for_action_completion(action,ip=config['amr']['Banana'])

        AMR.go_to(target_x=config['points'][point_name]['x'], target_y=config['points'][point_name]['y'], yaw=config['points'][point_name]['yaw'], speed_ratio=2.0)

        if point_name == "Table1":
            current_position = AMR.get_robot_pose()
            while True:
                distance, _ = compute_distance_and_angle(current_pose=current_position, target_pose=config['points'][point_name])
                print(f"AMR moving to {point_name}, distance remaining: {distance:.2f} m")
                if distance < 1.0:
                    break # within 1 meter
                current_position = AMR.get_robot_pose()

            AMR.set_speed_limit(0.2)

            print("Start move the arm now")
        #   Start move robot arm
            if (go_home(cam_pose)):
                self.log(f"✓ AMR moving to {point_name} position")
            else:
                self.log(f"✗ Failed to move to {point_name} position")

if __name__ == "__main__":
    root = tk.Tk()
    app = RobotGUI(root)
    root.mainloop()
