# gui/main_window.py
"""
Robot GUI main window
- English-only (ASCII) strings to avoid encoding/font issues
- Thread-safe Tkinter: ALL UI updates happen on the main thread via root.after()
- Proper shutdown: stop vision/amr/robot on GUI close
"""

import threading
import tkinter as tk
from tkinter import ttk, messagebox
import yaml

from config import gui_config, robot_config

# AMR import
try:
    import sys
    sys.path.insert(0, '/ws_job_msislab/amr_project/src/home_pc')
    from amr.msis_amr_motion import MSIS_PHOEBUS
    from amr.util import compute_distance_and_angle
    AMR_AVAILABLE = True
except ImportError as e:
    print(f"[AMR] Import failed: {e}")
    MSIS_PHOEBUS = None
    compute_distance_and_angle = None
    AMR_AVAILABLE = False

from .pose_manager import PoseManager


class RobotGUI:
    """Robot Control GUI"""

    def __init__(self, root, controllers):
        """
        Args:
            root: Tkinter root
            controllers: dict with robot/vision controllers
        """
        self.root = root
        self.root.title("Fairino Robot Control")
        self.root.geometry("960x640")  # 1200x800 -> 960x640 (80% scale)

        # Controllers
        self.robot_connector = controllers["robot_connector"]
        self.robot_state = controllers["robot_state"]
        self.robot_motion = controllers["robot_motion"]
        self.gripper = controllers["gripper_controller"]
        self.return_home = controllers.get("return_home")
        self.box_detector = controllers.get("box_detector")
        self.auto_pick_place = controllers.get("auto_pick_place")
        self.safe_pick_place = controllers.get("safe_pick_place")  # Manual stacking

        # Pose manager
        self.pose_manager = PoseManager()

        # AMR
        self.amr = None
        self.amr_config = None
        if AMR_AVAILABLE:
            try:
                with open(gui_config.AMR_CONFIG_PATH, "r") as stream:
                    self.amr_config = yaml.safe_load(stream)
                self.amr = MSIS_PHOEBUS(
                    ip=self.amr_config["amr"]["Banana"],
                    port=self.amr_config["amr"]["port"],
                )
                print("[AMR] Initialized successfully")
            except Exception as e:
                print(f"[AMR] Initialization failed: {e}")

        # Left container (scrollable) - 더 넓게
        left_container = ttk.Frame(self.root)
        left_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 4))  # 10,5 -> 8,4

        self.canvas = tk.Canvas(left_container)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(left_container, orient=tk.VERTICAL, command=self.canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.left_frame = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.left_frame, anchor=tk.NW)

        self.left_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # Close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # UI build
        self.create_ui()

    # -------------------------
    # Thread-safe UI helpers
    # -------------------------
    def ui(self, func, *args, **kwargs):
        """Run UI code on Tk main thread."""
        self.root.after(0, lambda: func(*args, **kwargs))

    def log(self, message: str):
        """Thread-safe log append (NO root.update here)."""
        def _append():
            try:
                self.output_text.insert(tk.END, f"{message}\n")
                self.output_text.see(tk.END)
            except Exception:
                pass
        self.ui(_append)

    def _safe_call(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            self.log(f"[ERROR] {e}")
            return None

    def _is_robot_connected(self) -> bool:
        """Best-effort check (supports various connector implementations)."""
        rc = self.robot_connector
        if hasattr(rc, "is_connected") and callable(getattr(rc, "is_connected")):
            return bool(rc.is_connected())
        if hasattr(rc, "connected"):
            return bool(getattr(rc, "connected"))
        if hasattr(rc, "client") and getattr(rc, "client") is not None:
            return True
        return False

    # -------------------------
    # UI creation
    # -------------------------
    def create_ui(self):
        self.create_connection_frame()
        self.create_control_frame()
        self.create_vision_frame()
        self.create_joint_frame()
        self.create_gripper_frame()
        self.create_pose_frame()
        self.create_save_frame()
        self.create_param_frame()
        self.create_run_frame()
        self.create_auto_control_frame()
        self.create_output_frame()

    def create_connection_frame(self):
        frame = ttk.LabelFrame(self.left_frame, text="Robot Connection", padding=10)
        frame.pack(fill=tk.X, padx=8, pady=4)

        ttk.Label(frame, text="Robot IP:").grid(row=0, column=0, sticky=tk.W)

        self.ip_var = tk.StringVar(value=getattr(self.robot_connector, "ip", ""))
        ttk.Entry(frame, textvariable=self.ip_var, width=16).grid(row=0, column=1, sticky=tk.W, padx=4)

        ttk.Button(frame, text="Connect", command=self.on_connect).grid(row=0, column=2, padx=4)
        ttk.Button(frame, text="Disconnect", command=self.on_disconnect).grid(row=0, column=3, padx=4)

        self.status_label = ttk.Label(frame, text="Status: Disconnected", foreground="red")
        self.status_label.grid(row=0, column=4, padx=8)

    def create_control_frame(self):
        frame = ttk.LabelFrame(self.left_frame, text="Robot Controls", padding=10)
        frame.pack(fill=tk.X, padx=8, pady=4)

        ttk.Button(frame, text="STOP", command=self.on_stop, width=12).pack(side=tk.LEFT, padx=4)
        ttk.Button(frame, text="PAUSE", command=self.on_pause, width=12).pack(side=tk.LEFT, padx=4)
        ttk.Button(frame, text="RESUME", command=self.on_resume, width=12).pack(side=tk.LEFT, padx=4)
        ttk.Button(frame, text="GO HOME", command=self.on_go_home, width=12).pack(side=tk.LEFT, padx=4)

        ttk.Button(frame, text="EXIT", command=self.on_close, width=10).pack(side=tk.RIGHT, padx=4)

        if AMR_AVAILABLE:
            ttk.Button(frame, text="AMR Table1", command=lambda: self.on_amr_goto("Table1"), width=12).pack(side=tk.LEFT, padx=4)
            ttk.Button(frame, text="AMR Table2", command=lambda: self.on_amr_goto("Table2"), width=12).pack(side=tk.LEFT, padx=4)
            ttk.Button(frame, text="AMR Stop", command=self.on_amr_stop, width=12).pack(side=tk.LEFT, padx=4)

    def create_vision_frame(self):
        frame = ttk.LabelFrame(self.left_frame, text="Vision Control (Camera + YOLO)", padding=10)
        frame.pack(fill=tk.X, padx=8, pady=4)

        self.btn_vision_on = ttk.Button(frame, text="Vision ON", command=self.on_vision_on, width=12)
        self.btn_vision_on.pack(side=tk.LEFT, padx=4)

        self.btn_vision_off = ttk.Button(frame, text="Vision OFF", command=self.on_vision_off, width=12, state=tk.DISABLED)
        self.btn_vision_off.pack(side=tk.LEFT, padx=4)

        self.vision_status_label = ttk.Label(frame, text="Status: OFF", foreground="red")
        self.vision_status_label.pack(side=tk.LEFT, padx=8)

    def create_joint_frame(self):
        frame = ttk.LabelFrame(self.left_frame, text=f"Joint Movement (+/- {gui_config.JOINT_INCREMENT} deg)", padding=10)
        frame.pack(fill=tk.X, padx=8, pady=4)

        for i in range(6):
            sub_frame = ttk.Frame(frame)
            sub_frame.pack(fill=tk.X, padx=4, pady=2)

            ttk.Label(sub_frame, text=f"Joint {i+1}:", width=10).pack(side=tk.LEFT)
            ttk.Button(sub_frame, text="<", width=3, command=lambda j=i: self.on_joint_move(j, -1)).pack(side=tk.LEFT, padx=2)
            ttk.Button(sub_frame, text=">", width=3, command=lambda j=i: self.on_joint_move(j, 1)).pack(side=tk.LEFT, padx=2)

    def create_gripper_frame(self):
        frame = ttk.LabelFrame(self.left_frame, text="Gripper Control", padding=10)
        frame.pack(fill=tk.X, padx=8, pady=4)

        ttk.Button(frame, text="Open", command=self.on_gripper_open, width=16).pack(side=tk.LEFT, padx=4)
        ttk.Button(frame, text="Close", command=self.on_gripper_close, width=16).pack(side=tk.LEFT, padx=4)

    def create_pose_frame(self):
        frame = ttk.LabelFrame(self.left_frame, text="Current Pose (Cartesian)", padding=10)
        frame.pack(fill=tk.X, padx=8, pady=4)

        self.pose_label = ttk.Label(frame, text="Not available", font=("Courier", 9))
        self.pose_label.pack()

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(pady=5)

        ttk.Button(btn_frame, text="Refresh Pose", command=self.on_refresh_pose).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="Save as Home", command=self.on_save_home).pack(side=tk.LEFT, padx=3)

    def create_save_frame(self):
        frame = ttk.LabelFrame(self.left_frame, text="Pose Management", padding=10)
        frame.pack(fill=tk.X, padx=8, pady=4)

        ttk.Button(frame, text="Save Pick", command=self.on_save_pick, width=12).pack(side=tk.LEFT, padx=3)
        ttk.Button(frame, text="Save Safe Pick", command=self.on_save_safe_pick, width=12).pack(side=tk.LEFT, padx=3)
        ttk.Button(frame, text="Save Place", command=self.on_save_place, width=12).pack(side=tk.LEFT, padx=3)
        ttk.Button(frame, text="Save Safe Place", command=self.on_save_safe_place, width=12).pack(side=tk.LEFT, padx=3)
        ttk.Button(frame, text="View Poses", command=self.on_view_poses, width=12).pack(side=tk.LEFT, padx=3)

    def create_param_frame(self):
        frame = ttk.LabelFrame(self.left_frame, text="Stacking Parameters", padding=10)
        frame.pack(fill=tk.X, padx=8, pady=4)

        ttk.Label(frame, text="Number of Boxes:").grid(row=0, column=0, sticky=tk.W, padx=4)
        self.num_boxes_var = tk.IntVar(value=self.pose_manager.get_param("num_boxes", 4))
        ttk.Entry(frame, textvariable=self.num_boxes_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=4)

        ttk.Label(frame, text="Box Height (mm):").grid(row=0, column=2, sticky=tk.W, padx=4)
        self.box_height_var = tk.DoubleVar(value=self.pose_manager.get_param("box_height", 58.0))
        ttk.Entry(frame, textvariable=self.box_height_var, width=10).grid(row=0, column=3, sticky=tk.W, padx=4)

        ttk.Label(frame, text="Table:").grid(row=0, column=4, sticky=tk.W, padx=4)
        self.table_var = tk.StringVar(value="1")
        self.table_combo = ttk.Combobox(frame, textvariable=self.table_var, values=["1", "2", "3", "4", "5"], width=5, state="readonly")
        self.table_combo.grid(row=0, column=5, sticky=tk.W, padx=4)

        ttk.Label(frame, text="Normal Speed:").grid(row=1, column=0, sticky=tk.W, padx=4)
        self.vel_normal_var = tk.DoubleVar(value=self.pose_manager.get_param("vel_normal", 30.0))
        ttk.Entry(frame, textvariable=self.vel_normal_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=4)

        ttk.Label(frame, text="Slow Speed:").grid(row=1, column=2, sticky=tk.W, padx=4)
        self.vel_slow_var = tk.DoubleVar(value=self.pose_manager.get_param("vel_slow", 30.0))
        ttk.Entry(frame, textvariable=self.vel_slow_var, width=10).grid(row=1, column=3, sticky=tk.W, padx=4)

        ttk.Label(frame, text="Approach Offset (mm):").grid(row=2, column=0, sticky=tk.W, padx=4)
        self.approach_offset_var = tk.DoubleVar(value=self.pose_manager.get_param("approach_offset", 40.0))
        ttk.Entry(frame, textvariable=self.approach_offset_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=4)

        ttk.Label(frame, text="Push Direction:").grid(row=2, column=2, sticky=tk.W, padx=4)
        self.push_direction_var = tk.StringVar(value=self.pose_manager.get_param("push_direction", "LEFT"))
        self.push_combo = ttk.Combobox(frame, textvariable=self.push_direction_var, values=["LEFT", "RIGHT"], width=8, state="readonly")
        self.push_combo.grid(row=2, column=3, sticky=tk.W, padx=4)

        ttk.Label(frame, text="Column Offset (mm):").grid(row=2, column=4, sticky=tk.W, padx=4)
        self.column_offset_var = tk.DoubleVar(value=self.pose_manager.get_param("column_offset", 100.0))
        ttk.Entry(frame, textvariable=self.column_offset_var, width=10).grid(row=2, column=5, sticky=tk.W, padx=4)

        ttk.Button(frame, text="Save Parameters", command=self.on_save_params).grid(row=1, column=4, columnspan=2, sticky=tk.W, padx=4)

    def create_run_frame(self):
        frame = ttk.LabelFrame(self.left_frame, text="Stacking Control (Manual)", padding=10)
        frame.pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(frame, text="Start Stacking (Manual)", command=self.on_start_stacking, width=25).pack(side=tk.LEFT, padx=4)

    def create_auto_control_frame(self):
        frame = ttk.LabelFrame(self.left_frame, text="Auto Pick and Place", padding=10)
        frame.pack(fill=tk.X, padx=8, pady=4)

        counter_frame = ttk.Frame(frame)
        counter_frame.pack(fill=tk.X, pady=5)

        ttk.Label(counter_frame, text="Stack Counter:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=4)
        self.stack_counter_label = ttk.Label(counter_frame, text="0", font=("Arial", 12, "bold"), foreground="blue")
        self.stack_counter_label.pack(side=tk.LEFT, padx=4)

        ttk.Button(counter_frame, text="Reset Counter", command=self.on_reset_counter, width=12).pack(side=tk.LEFT, padx=15)

        ttk.Separator(frame, orient="horizontal").pack(fill=tk.X, pady=10)

        loop_frame = ttk.Frame(frame)
        loop_frame.pack(fill=tk.X, pady=5)

        ttk.Label(loop_frame, text="12 Auto Loop:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=4)
        ttk.Label(loop_frame, text="boxes:").pack(side=tk.LEFT, padx=4)
        self.auto_boxes_var = tk.IntVar(value=4)
        ttk.Spinbox(loop_frame, from_=1, to=20, textvariable=self.auto_boxes_var, width=10).pack(side=tk.LEFT, padx=4)

        ttk.Button(loop_frame, text="Start Auto Loop", command=self.on_auto_loop, width=18).pack(side=tk.LEFT, padx=8)

        quick_frame = ttk.Frame(frame)
        quick_frame.pack(fill=tk.X, pady=5)

        ttk.Label(quick_frame, text="13 Quick Start:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=4)
        ttk.Label(quick_frame, text="auto reset + 4 boxes").pack(side=tk.LEFT, padx=4)
        ttk.Button(quick_frame, text="Quick Start", command=self.on_quick_start, width=18).pack(side=tk.LEFT, padx=8)

    def create_output_frame(self):
        # 우측 프레임 - 고정 폭으로 설정
        frame = ttk.LabelFrame(self.root, text="Status Output", padding=10)
        frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 10), pady=5)

        self.output_text = tk.Text(frame, height=45, width=45, font=("Courier", 8))
        self.output_text.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.output_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.config(yscrollcommand=scrollbar.set)

    # -------------------------
    # Event handlers
    # -------------------------
    def on_connect(self):
        ip = self.ip_var.get().strip()
        if hasattr(self.robot_connector, "ip"):
            self.robot_connector.ip = ip

        if self.robot_connector.connect():
            self.status_label.config(text="Status: Connected", foreground="green")
            self.log("Robot connected")
        else:
            self.log("Connection failed")
            messagebox.showerror("Error", "Connection failed")

    def on_disconnect(self):
        if self.robot_connector.disconnect():
            self.status_label.config(text="Status: Disconnected", foreground="red")
            self.log("Robot disconnected")

    def on_stop(self):
        self.robot_connector.stop_motion()
        self.log("STOP")

    def on_pause(self):
        self.robot_connector.pause_motion()
        self.log("PAUSE")

    def on_resume(self):
        self.robot_connector.resume_motion()
        self.log("RESUME")

    def on_go_home(self):
        def home_thread():
            # GO HOME = CAM_POSE로 이동 (비전 확인용)
            self.log("Moving to CAM_POSE (vision position)...")
            err = self.robot_motion.move_cart(gui_config.CAM_POSE)
            if err == 0:
                self.log("Arrived at CAM_POSE")
            else:
                self.log("Failed to move to CAM_POSE")
        threading.Thread(target=home_thread, daemon=True).start()

    def on_joint_move(self, joint_index: int, direction: int):
        def joint_thread():
            delta = direction * gui_config.JOINT_INCREMENT
            err = self.robot_motion.rotate_single_joint(joint_index, delta)
            if err == 0:
                self.log(f"J{joint_index+1} moved {delta:+.1f} deg")
            else:
                self.log(f"J{joint_index+1} move failed")
        threading.Thread(target=joint_thread, daemon=True).start()

    def on_gripper_open(self):
        self.gripper.open()
        self.log("Gripper opened")

    def on_gripper_close(self):
        self.gripper.close()
        self.log("Gripper closed")

    def on_refresh_pose(self):
        (e_p, pose), (e_j, joint) = self.robot_state.read_pose_joint()
        if e_p == 0 and pose:
            pose_str = f"[{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}, {pose[3]:.3f}, {pose[4]:.3f}, {pose[5]:.3f}]"
            self.pose_label.config(text=pose_str)
            self.log(f"Current pose: {pose_str}")
        else:
            self.log("Failed to read pose")

    def on_save_home(self):
        (e_p, pose), (e_j, joint) = self.robot_state.read_pose_joint()
        if e_j == 0 and joint:
            self.robot_state.set_initial_joint6(joint, force=True)  # 강제로 덮어쓰기
            self.log(f"Home saved: {self.robot_state.fmt_joint(joint)}")
            messagebox.showinfo("Success", "Home position saved.")
        else:
            self.log("Failed to save home")
            messagebox.showerror("Error", "Could not read current position")

    def on_save_pick(self):
        self._save_pose("pick_pose", "Pick")

    def on_save_safe_pick(self):
        self._save_pose("safe_pick", "Safe Pick")

    def on_save_place(self):
        self._save_pose("place_pose", "Place")

    def on_save_safe_place(self):
        self._save_pose("safe_place", "Safe Place")

    def _save_pose(self, pose_type: str, label: str):
        (e_p, pose), (e_j, joint) = self.robot_state.read_pose_joint()
        if e_p == 0 and pose:
            table = self.table_var.get()
            self.pose_manager.set_table_pose(table, pose_type, pose)
            if self.pose_manager.save():
                self.log(f"{label} saved (Table {table})")
                messagebox.showinfo("Success", f"{label} saved for Table {table}")
            else:
                self.log(f"Failed to save {label}")
                messagebox.showerror("Error", f"Failed to save {label}")
        else:
            messagebox.showerror("Error", "Could not get current pose")

    def on_view_poses(self):
        table = self.table_var.get()
        poses = self.pose_manager.get_table_poses(table)

        self.log("=== Saved Poses ===")
        self.log(f"Table {table}:")
        self.log(f"Pick: {poses.get('pick_pose')}")
        self.log(f"Safe Pick: {poses.get('safe_pick')}")
        self.log(f"Place: {poses.get('place_pose')}")
        self.log(f"Safe Place: {poses.get('safe_place')}")

    def on_save_params(self):
        try:
            self.pose_manager.set_param("num_boxes", self.num_boxes_var.get())
            self.pose_manager.set_param("box_height", self.box_height_var.get())
            self.pose_manager.set_param("vel_normal", self.vel_normal_var.get())
            self.pose_manager.set_param("vel_slow", self.vel_slow_var.get())
            self.pose_manager.set_param("approach_offset", self.approach_offset_var.get())
            self.pose_manager.set_param("push_direction", self.push_direction_var.get())
            self.pose_manager.set_param("column_offset", self.column_offset_var.get())

            if self.pose_manager.save():
                self.log("Parameters saved")
                messagebox.showinfo("Success", "Parameters saved.")
            else:
                self.log("Failed to save parameters")
                messagebox.showerror("Error", "Failed to save parameters.")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid parameters: {e}")

    def on_start_stacking(self):
        """Manual stacking using saved poses"""
        if not self.safe_pick_place:
            self.log("SafePickPlace not initialized")
            messagebox.showerror("Error", "Manual stacking module not available")
            return

        table = self.table_var.get()
        poses = self.pose_manager.get_table_poses(table)

        pick_pose = poses.get("pick_pose")
        place_pose = poses.get("place_pose")
        safe_pick = poses.get("safe_pick")
        safe_place = poses.get("safe_place")

        if not all([pick_pose, place_pose, safe_pick, safe_place]):
            messagebox.showerror("Error", f"Table {table} poses not saved.\nPlease save all 4 poses first.")
            self.log(f"Table {table} poses incomplete")
            return

        num_boxes = int(self.num_boxes_var.get())
        box_height = float(self.box_height_var.get())
        vel_normal = float(self.vel_normal_var.get())
        vel_slow = float(self.vel_slow_var.get())
        approach_offset = float(self.approach_offset_var.get())
        push_direction = self.push_direction_var.get()
        column_offset = float(self.column_offset_var.get())

        total_boxes = num_boxes * 2  # 2 columns
        
        answer = messagebox.askyesno(
            "Manual Stacking",
            f"Start manual stacking?\n\n"
            f"Table: {table}\n"
            f"Boxes per Column: {num_boxes}\n"
            f"Total Boxes: {total_boxes}\n"
            f"Height: {box_height} mm\n"
            f"Speed: {vel_normal}/{vel_slow}\n"
            f"Approach: {approach_offset} mm\n"
            f"Column Offset: {column_offset} mm\n"
            f"Push: {push_direction}"
        )
        if not answer:
            return

        def stacking_thread():
            self.log("=" * 50)
            self.log(f"Start Manual Stacking (Table {table})")
            self.log("=" * 50)

            try:
                result = self.safe_pick_place.run_stacking_sequence(
                    pick_pose=pick_pose,
                    place_pose=place_pose,
                    safe_pick=safe_pick,
                    safe_place=safe_place,
                    num_boxes=num_boxes,
                    box_height=box_height,
                    approach_offset=approach_offset,
                    push_direction=push_direction,
                    column_offset=column_offset,
                    vel_normal=vel_normal,
                    vel_slow=vel_slow,
                    status_callback=self.log
                )
            except Exception as e:
                result = {"ok": False, "boxes_moved": 0}
                self.log(f"Error: {e}")

            if result.get("ok"):
                moved = result.get("boxes_moved", 0)
                total = result.get("total_requested", num_boxes)
                self.log("=" * 50)
                self.log(f"Manual stacking done ({moved}/{total} boxes)")
                self.log("=" * 50)
                self.ui(messagebox.showinfo, "Success", f"Stacking done.\nMoved: {moved}/{total}")
            else:
                self.log("Manual stacking failed")
                self.ui(messagebox.showerror, "Error", "Manual stacking failed")

        threading.Thread(target=stacking_thread, daemon=True).start()

    # -------------------------
    # Vision control
    # -------------------------
    def on_vision_on(self):
        if not self.box_detector:
            self.log("BoxDetector not initialized")
            messagebox.showerror("Error", "Vision module not available")
            return

        self.log("[Vision] Starting camera and YOLO...")

        def vision_start_thread():
            success = False
            try:
                success = bool(self.box_detector.start())
            except Exception as e:
                self.log(f"[Vision] Start failed: {e}")
                success = False

            if success:
                self.ui(self.vision_status_label.config, text="Status: ON", foreground="green")
                self.ui(self.btn_vision_on.config, state=tk.DISABLED)
                self.ui(self.btn_vision_off.config, state=tk.NORMAL)
                self.log("Vision started")
            else:
                self.log("Vision start failed")
                self.ui(messagebox.showerror, "Error", "Failed to start vision")

        threading.Thread(target=vision_start_thread, daemon=True).start()

    def on_vision_off(self):
        if not self.box_detector:
            return

        self.log("[Vision] Stopping...")

        def vision_stop_thread():
            try:
                self.box_detector.stop()
            except Exception as e:
                self.log(f"[Vision] Stop error: {e}")

            self.ui(self.vision_status_label.config, text="Status: OFF", foreground="red")
            self.ui(self.btn_vision_on.config, state=tk.NORMAL)
            self.ui(self.btn_vision_off.config, state=tk.DISABLED)
            self.log("Vision stopped")

        threading.Thread(target=vision_stop_thread, daemon=True).start()

    # -------------------------
    # Auto Loop (12)
    # -------------------------
    def on_auto_loop(self):
        if not self.auto_pick_place:
            self.log("AutoPickPlace not initialized")
            messagebox.showerror("Error", "Auto pick and place module not available")
            return

        num_boxes = int(self.auto_boxes_var.get())

        answer = messagebox.askyesno(
            "Auto Loop (12)",
            f"Run auto loop for {num_boxes} boxes?\n\n"
            "- Robot connection required\n"
            "- Vision ON required"
        )
        if not answer:
            return

        def auto_loop_thread():
            self.log("=" * 50)
            self.log(f"Start Auto Loop (12) - {num_boxes} boxes")
            self.log("=" * 50)

            try:
                result = self.auto_pick_place.cmd12_auto_loop(num_cycles=num_boxes)
            except Exception as e:
                result = {"ok": False, "msg": str(e)}

            if result.get("ok"):
                counter = result.get("stack_counter", 0)
                self.ui(self.update_stack_counter, counter)
                self.log("=" * 50)
                self.log(f"Auto Loop done (total moved: {counter})")
                self.log("=" * 50)
                self.ui(messagebox.showinfo, "Success", f"Auto Loop done.\nTotal moved: {counter}")
            else:
                msg = result.get("msg", "Unknown error")
                self.log(f"Auto Loop failed: {msg}")
                self.ui(messagebox.showerror, "Error", f"Auto Loop failed:\n{msg}")

        threading.Thread(target=auto_loop_thread, daemon=True).start()

    # -------------------------
    # Quick Start (13) - Full Auto Sequence
    # -------------------------
    def on_quick_start(self):
        """
        전체 자동화 시퀀스:
        1. AMR Table 1 버튼 클릭 (이동)
        2. Home 저장
        3. 박스 6개 옮기기
        4. AMR Table 2 버튼 클릭 (이동)
        5. Start Stacking 실행
        """
        if not self.auto_pick_place:
            self.log("AutoPickPlace not initialized")
            messagebox.showerror("Error", "Auto pick and place module not available")
            return
        
        if not self.safe_pick_place:
            self.log("SafePickPlace not initialized")
            messagebox.showerror("Error", "Manual stacking module not available")
            return

        answer = messagebox.askyesno(
            "Quick Start - Full Auto (13)",
            "전체 자동화 시퀀스를 실행하시겠습니까?\n\n"
            "1️⃣ AMR Table 1로 이동\n"
            "2️⃣ Home 저장\n"
            "3️⃣ 박스 6개 옮기기 (Vision Pick & Place)\n"
            "4️⃣ AMR Table 2로 이동\n"
            "5️⃣ Manual Stacking 실행\n"
        )
        if not answer:
            return

        def full_auto_sequence():
            self.log("=" * 60)
            self.log("🚀 Full Auto Sequence Start")
            self.log("=" * 60)
            
            try:
                # ========== STEP 1: AMR Table 1 버튼 클릭 ==========
                self.log("\n" + "=" * 60)
                self.log("📍 STEP 1: AMR Table 1 버튼 실행...")
                self.log("=" * 60)
                
                # AMR Table 1 이동 (기존 버튼 기능 호출)
                self.on_amr_goto("Table1")
                
                # AMR 이동 완료 대기 (간단한 방법)
                import time
                time.sleep(2)  # AMR 이동 시작 대기
                
                self.log("✅ AMR Table 1 이동 완료")
                
                # ========== STEP 2: Home 저장 ==========
                self.log("\n" + "=" * 60)
                self.log("💾 STEP 2: Saving Home Position...")
                self.log("=" * 60)
                
                joint = self.robot_state.get_current_joint()
                if joint is None:
                    raise Exception("현재 Joint를 가져올 수 없습니다")
                
                self.robot_state.set_initial_joint6(joint)
                self.log(f"✅ Home saved: {joint}")
                
                # ========== STEP 3: 박스 6개 옮기기 ==========
                self.log("\n" + "=" * 60)
                self.log("📦 STEP 3: Auto Pick & Place (6 boxes)...")
                self.log("=" * 60)
                
                result = self.auto_pick_place.cmd12_auto_loop(num_cycles=6)
                
                if not result.get("ok"):
                    msg = result.get("msg", "Unknown error")
                    raise Exception(f"Auto Pick & Place 실패: {msg}")
                
                counter = result.get("stack_counter", 0)
                self.ui(self.update_stack_counter, counter)
                self.log(f"✅ Auto Pick & Place 완료 (총 {counter}개)")
                
                # ========== STEP 4: AMR Table 2 버튼 클릭 ==========
                self.log("\n" + "=" * 60)
                self.log("📍 STEP 4: AMR Table 2 버튼 실행...")
                self.log("=" * 60)
                
                # AMR Table 2 이동 (기존 버튼 기능 호출)
                self.on_amr_goto("Table2")
                
                # AMR 이동 완료 대기
                time.sleep(2)
                
                self.log("✅ AMR Table 2 이동 완료")
                
                # ========== STEP 5: Manual Stacking 실행 ==========
                self.log("\n" + "=" * 60)
                self.log("🏗️ STEP 5: Manual Stacking...")
                self.log("=" * 60)
                
                # Table 2 포즈 가져오기
                table2_poses = self.pose_manager.get_table_poses("2")
                pick_pose = table2_poses.get("pick_pose")
                place_pose = table2_poses.get("place_pose")
                safe_pick = table2_poses.get("safe_pick")
                safe_place = table2_poses.get("safe_place")
                
                if not all([pick_pose, place_pose, safe_pick, safe_place]):
                    raise Exception("Table 2의 포즈가 모두 저장되지 않았습니다")
                
                # 설정 가져오기
                num_boxes = int(self.num_boxes_var.get())
                box_height = float(self.box_height_var.get())
                vel_normal = float(self.vel_normal_var.get())
                vel_slow = float(self.vel_slow_var.get())
                approach_offset = float(self.approach_offset_var.get())
                push_direction = self.push_direction_var.get()
                column_offset = float(self.column_offset_var.get())
                
                self.log(f"Stacking parameters: {num_boxes} boxes/column, {box_height}mm height")
                
                stacking_result = self.safe_pick_place.run_stacking_sequence(
                    pick_pose=pick_pose,
                    place_pose=place_pose,
                    safe_pick=safe_pick,
                    safe_place=safe_place,
                    num_boxes=num_boxes,
                    box_height=box_height,
                    approach_offset=approach_offset,
                    push_direction=push_direction,
                    column_offset=column_offset,
                    vel_normal=vel_normal,
                    vel_slow=vel_slow,
                    status_callback=self.log
                )
                
                if not stacking_result.get("ok"):
                    raise Exception("Manual Stacking 실패")
                
                moved = stacking_result.get("boxes_moved", 0)
                self.log(f"✅ Manual Stacking 완료 (총 {moved}개)")
                
                # ========== 완료 ==========
                self.log("\n" + "=" * 60)
                self.log("🎉 Full Auto Sequence Complete!")
                self.log(f"📊 총 Pick & Place: {counter}개")
                self.log(f"📊 총 Manual Stack: {moved}개")
                self.log("=" * 60)
                
                self.ui(messagebox.showinfo, 
                       "Success", 
                       f"전체 자동화 완료!\n\n"
                       f"✅ Vision Pick & Place: {counter}개\n"
                       f"✅ Manual Stacking: {moved}개")
                
            except Exception as e:
                self.log("\n" + "=" * 60)
                self.log(f"❌ Full Auto Sequence Failed: {str(e)}")
                self.log("=" * 60)
                self.ui(messagebox.showerror, "Error", f"자동화 실패:\n\n{str(e)}")

        threading.Thread(target=full_auto_sequence, daemon=True).start()

    # -------------------------
    # Counter
    # -------------------------
    def update_stack_counter(self, count: int):
        self.stack_counter_label.config(text=str(int(count)))
        self.log(f"Stack Counter: {count}")

    def on_reset_counter(self):
        if not self.auto_pick_place:
            self.log("AutoPickPlace not initialized")
            return

        answer = messagebox.askyesno("Reset Counter", "Reset stack counter to 0?")
        if answer:
            try:
                self.auto_pick_place.reset_stack_counter()
            except Exception as e:
                self.log(f"Reset counter error: {e}")
            self.update_stack_counter(0)
            self.log("Stack counter reset to 0")

    # -------------------------
    # AMR control
    # -------------------------
    def on_amr_goto(self, point_name: str):
        if not self.amr or not self.amr_config:
            self.log("AMR not initialized")
            return

        def amr_thread():
            try:
                self.log(f"[AMR] Moving to {point_name}...")
                self.amr.set_speed_limit(5.0)

                if point_name == "Table1":
                    action = self.amr.go_to(
                        target_x=self.amr_config["points"]["Rotate1"]["x"],
                        target_y=self.amr_config["points"]["Rotate1"]["y"],
                        yaw=self.amr_config["points"]["Rotate1"]["yaw"],
                        speed_ratio=2.0,
                    )
                    self.amr._wait_for_action_completion(action, ip=self.amr_config["amr"]["Banana"])

                self.amr.go_to(
                    target_x=self.amr_config["points"][point_name]["x"],
                    target_y=self.amr_config["points"][point_name]["y"],
                    yaw=self.amr_config["points"][point_name]["yaw"],
                    speed_ratio=2.0,
                )

                if point_name == "Table1" and compute_distance_and_angle is not None:
                    current_position = self.amr.get_robot_pose()
                    while True:
                        distance, _ = compute_distance_and_angle(
                            current_pose=current_position,
                            target_pose=self.amr_config["points"][point_name],
                        )
                        self.log(f"[AMR] Distance: {distance:.2f} m")
                        if distance < 1.0:
                            break
                        current_position = self.amr.get_robot_pose()

                    self.amr.set_speed_limit(0.2)

                self.log(f"[AMR] Arrived at {point_name}")

                self.log("[AMR] Moving robot arm to camera pose...")
                self.robot_motion.move_cart(gui_config.CAM_POSE)
                self.log("Robot arm ready")

            except Exception as e:
                self.log(f"[AMR] Error: {e}")

        threading.Thread(target=amr_thread, daemon=True).start()

    def on_amr_stop(self):
        if self.amr:
            try:
                self.amr.stop_motion_now()
                self.log("[AMR] Stopped")
            except Exception as e:
                self.log(f"[AMR] Stop failed: {e}")

    # -------------------------
    # Close / Shutdown
    # -------------------------
    def on_close(self):
        """Close GUI and stop everything safely."""
        answer = messagebox.askyesno("Exit", "Exit the GUI and stop modules?")
        if not answer:
            return

        self.log("[SYS] Shutting down...")

        # Stop vision
        if self.box_detector:
            try:
                self.box_detector.stop()
                self.log("[SYS] Vision stopped")
            except Exception as e:
                self.log(f"[SYS] Vision stop error: {e}")

        # Stop AMR
        if self.amr:
            try:
                self.amr.stop_motion_now()
                self.log("[SYS] AMR stopped")
            except Exception as e:
                self.log(f"[SYS] AMR stop error: {e}")

        # Stop auto controllers
        if self.auto_pick_place:
            for name in ["stop", "cancel", "request_stop", "shutdown"]:
                if hasattr(self.auto_pick_place, name) and callable(getattr(self.auto_pick_place, name)):
                    try:
                        getattr(self.auto_pick_place, name)()
                        self.log(f"[SYS] AutoPickPlace {name}() called")
                    except Exception as e:
                        self.log(f"[SYS] AutoPickPlace {name}() error: {e}")
                    break

        if self.safe_pick_place:
            try:
                self.safe_pick_place.request_stop()
                self.log("[SYS] SafePickPlace stopped")
            except Exception as e:
                self.log(f"[SYS] SafePickPlace stop error: {e}")

        # Disconnect robot
        try:
            self.robot_connector.disconnect()
            self.log("[SYS] Robot disconnected")
        except Exception as e:
            self.log(f"[SYS] Robot disconnect error: {e}")

        # Destroy UI
        try:
            self.root.destroy()
        except Exception:
            pass

    # -------------------------
    # Misc
    # -------------------------
    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")