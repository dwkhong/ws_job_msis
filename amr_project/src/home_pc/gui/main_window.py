# gui/main_window.py
"""
Robot GUI 메인 윈도우
- 원본 GUI 구조 유지
- 백엔드는 우리 클래스 사용
"""
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import yaml

from config import gui_config, robot_config

# AMR import
try:
    import sys
    sys.path.insert(0, '/home/dw/ws_job_msislab/amr_project/src/home_pc')
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
        self.root.geometry("900x800")
        
        # Controllers
        self.robot_connector = controllers['robot_connector']
        self.robot_state = controllers['robot_state']
        self.robot_motion = controllers['robot_motion']
        self.gripper = controllers['gripper_controller']
        self.return_home = controllers.get('return_home')
        self.box_detector = controllers.get('box_detector')  # NEW!
        self.auto_pick_place = controllers.get('auto_pick_place')  # NEW!
        
        # Pose 관리
        self.pose_manager = PoseManager()
        
        # AMR 초기화
        self.amr = None
        self.amr_config = None
        if AMR_AVAILABLE:
            try:
                with open(gui_config.AMR_CONFIG_PATH, 'r') as stream:
                    self.amr_config = yaml.safe_load(stream)
                self.amr = MSIS_PHOEBUS(
                    ip=self.amr_config['amr']['Banana'],
                    port=self.amr_config['amr']['port']
                )
                print("[AMR] Initialized successfully")
            except Exception as e:
                print(f"[AMR] Initialization failed: {e}")
        
        # 좌측 프레임 (스크롤 가능)
        # Canvas + Scrollbar 구조
        left_container = ttk.Frame(self.root)
        left_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Canvas
        self.canvas = tk.Canvas(left_container)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(left_container, orient=tk.VERTICAL, command=self.canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # 실제 컨텐츠가 들어갈 프레임
        self.left_frame = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.left_frame, anchor=tk.NW)
        
        # 스크롤 영역 업데이트
        self.left_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        
        # 마우스 휠 스크롤
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # UI 생성
        self.create_ui()
    
    def create_ui(self):
        """UI 생성"""
        # 좌측 프레임은 이미 __init__에서 생성됨
        
        # 1. Connection
        self.create_connection_frame()
        
        # 2. Controls
        self.create_control_frame()
        
        # 3. Vision Control (NEW!)
        self.create_vision_frame()
        
        # 4. Joint Movement
        self.create_joint_frame()
        
        # 5. Gripper
        self.create_gripper_frame()
        
        # 6. Current Pose
        self.create_pose_frame()
        
        # 7. Pose Management
        self.create_save_frame()
        
        # 8. Stacking Parameters
        self.create_param_frame()
        
        # 9. Stacking Control (Original)
        self.create_run_frame()
        
        # 10. Auto Control (12번)
        self.create_auto_control_frame()
        
        # 우측 프레임 (Output)
        self.create_output_frame()
    
    def create_connection_frame(self):
        """연결 프레임"""
        frame = ttk.LabelFrame(self.left_frame, text="Robot Connection", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(frame, text="Robot IP:").grid(row=0, column=0, sticky=tk.W)
        
        self.ip_var = tk.StringVar(value=self.robot_connector.ip)
        ttk.Entry(frame, textvariable=self.ip_var, width=20).grid(row=0, column=1, sticky=tk.W, padx=5)
        
        ttk.Button(frame, text="Connect", command=self.on_connect).grid(row=0, column=2, padx=5)
        ttk.Button(frame, text="Disconnect", command=self.on_disconnect).grid(row=0, column=3, padx=5)
        
        self.status_label = ttk.Label(frame, text="Status: Disconnected", foreground="red")
        self.status_label.grid(row=0, column=4, padx=10)
    
    def create_control_frame(self):
        """제어 프레임"""
        frame = ttk.LabelFrame(self.left_frame, text="Robot Controls", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(frame, text="STOP", command=self.on_stop, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="PAUSE", command=self.on_pause, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="RESUME", command=self.on_resume, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="GO HOME", command=self.on_go_home, width=15).pack(side=tk.LEFT, padx=5)
        
        # AMR 버튼 (사용 가능할 때만)
        if AMR_AVAILABLE:
            ttk.Button(frame, text="AMR table1", command=lambda: self.on_amr_goto("Table1"), width=10).pack(side=tk.LEFT, padx=5)
            ttk.Button(frame, text="AMR table2", command=lambda: self.on_amr_goto("Table2"), width=10).pack(side=tk.LEFT, padx=5)
            ttk.Button(frame, text="AMR stop", command=self.on_amr_stop, width=10).pack(side=tk.LEFT, padx=5)
    
    def create_vision_frame(self):
        """Vision 제어 프레임 (NEW!)"""
        frame = ttk.LabelFrame(self.left_frame, text="Vision Control (Camera + YOLO)", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Vision On/Off 버튼
        self.btn_vision_on = ttk.Button(frame, text="📷 Vision ON", command=self.on_vision_on, width=15)
        self.btn_vision_on.pack(side=tk.LEFT, padx=5)
        
        self.btn_vision_off = ttk.Button(frame, text="📷 Vision OFF", command=self.on_vision_off, width=15, state=tk.DISABLED)
        self.btn_vision_off.pack(side=tk.LEFT, padx=5)
        
        # Vision 상태 표시
        self.vision_status_label = ttk.Label(frame, text="Status: OFF", foreground="red")
        self.vision_status_label.pack(side=tk.LEFT, padx=10)
    
    def create_joint_frame(self):
        """Joint 이동 프레임"""
        frame = ttk.LabelFrame(self.left_frame, text=f"Joint Movement (±{gui_config.JOINT_INCREMENT}°)", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        for i in range(6):
            sub_frame = ttk.Frame(frame)
            sub_frame.pack(fill=tk.X, padx=5, pady=3)
            
            ttk.Label(sub_frame, text=f"Joint {i+1}:", width=10).pack(side=tk.LEFT)
            ttk.Button(sub_frame, text="←", width=3, 
                      command=lambda j=i: self.on_joint_move(j, -1)).pack(side=tk.LEFT, padx=2)
            ttk.Button(sub_frame, text="→", width=3, 
                      command=lambda j=i: self.on_joint_move(j, 1)).pack(side=tk.LEFT, padx=2)
    
    def create_gripper_frame(self):
        """그리퍼 프레임"""
        frame = ttk.LabelFrame(self.left_frame, text="Gripper Control", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(frame, text="Open Gripper", command=self.on_gripper_open, width=20).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="Close Gripper", command=self.on_gripper_close, width=20).pack(side=tk.LEFT, padx=5)
    
    def create_pose_frame(self):
        """현재 Pose 프레임"""
        frame = ttk.LabelFrame(self.left_frame, text="Current Pose (Cartesian)", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.pose_label = ttk.Label(frame, text="Not available", font=("Courier", 9))
        self.pose_label.pack()
        
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(pady=5)
        
        ttk.Button(btn_frame, text="Refresh Pose", command=self.on_refresh_pose).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="💾 Save as Home", command=self.on_save_home).pack(side=tk.LEFT, padx=3)
    
    def create_save_frame(self):
        """Pose 저장 프레임"""
        frame = ttk.LabelFrame(self.left_frame, text="Pose Management", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(frame, text="Save Pick Pose", command=self.on_save_pick, width=18).pack(side=tk.LEFT, padx=3)
        ttk.Button(frame, text="Save Safe Pick", command=self.on_save_safe_pick, width=18).pack(side=tk.LEFT, padx=3)
        ttk.Button(frame, text="Save Place Pose", command=self.on_save_place, width=18).pack(side=tk.LEFT, padx=3)
        ttk.Button(frame, text="Save Safe Place", command=self.on_save_safe_place, width=18).pack(side=tk.LEFT, padx=3)
        ttk.Button(frame, text="View Saved Poses", command=self.on_view_poses, width=18).pack(side=tk.LEFT, padx=3)
    
    def create_param_frame(self):
        """스택 파라미터 프레임"""
        frame = ttk.LabelFrame(self.left_frame, text="Stacking Parameters", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Row 1
        ttk.Label(frame, text="Number of Boxes:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.num_boxes_var = tk.IntVar(value=self.pose_manager.get_param("num_boxes", 4))
        ttk.Entry(frame, textvariable=self.num_boxes_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(frame, text="Box Height (mm):").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.box_height_var = tk.DoubleVar(value=self.pose_manager.get_param("box_height", 58.0))
        ttk.Entry(frame, textvariable=self.box_height_var, width=10).grid(row=0, column=3, sticky=tk.W, padx=5)
        
        ttk.Label(frame, text="Table:").grid(row=0, column=4, sticky=tk.W, padx=5)
        self.table_var = tk.StringVar(value="1")
        self.table_combo = ttk.Combobox(frame, textvariable=self.table_var, 
                                        values=["1","2","3","4","5"], width=5, state="readonly")
        self.table_combo.grid(row=0, column=5, sticky=tk.W, padx=5)
        
        # Row 2
        ttk.Label(frame, text="Normal Speed:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.vel_normal_var = tk.DoubleVar(value=self.pose_manager.get_param("vel_normal", 30.0))
        ttk.Entry(frame, textvariable=self.vel_normal_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(frame, text="Slow Speed:").grid(row=1, column=2, sticky=tk.W, padx=5)
        self.vel_slow_var = tk.DoubleVar(value=self.pose_manager.get_param("vel_slow", 30.0))
        ttk.Entry(frame, textvariable=self.vel_slow_var, width=10).grid(row=1, column=3, sticky=tk.W, padx=5)
        
        ttk.Button(frame, text="Save Parameters", command=self.on_save_params).grid(row=1, column=4, columnspan=2, sticky=tk.W, padx=5)
    
    def create_run_frame(self):
        """스택 실행 프레임"""
        frame = ttk.LabelFrame(self.left_frame, text="Stacking Control (Original)", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(frame, text="Start Stacking (Manual)", 
                  command=self.on_start_stacking, width=25).pack(side=tk.LEFT, padx=5)
    
    def create_auto_control_frame(self):
        """자동 제어 프레임 (12번, 13번)"""
        frame = ttk.LabelFrame(self.left_frame, text="🤖 Auto Pick & Place", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 스택 카운터 표시 및 리셋
        counter_frame = ttk.Frame(frame)
        counter_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(counter_frame, text="📦 Stack Counter:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        self.stack_counter_label = ttk.Label(counter_frame, text="0", 
                                             font=("Arial", 12, "bold"), foreground="blue")
        self.stack_counter_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(counter_frame, text="🔄 Reset Counter", 
                  command=self.on_reset_counter, width=15).pack(side=tk.LEFT, padx=15)
        
        # 구분선
        ttk.Separator(frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # 12번 - Custom Auto Loop
        loop_frame = ttk.Frame(frame)
        loop_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(loop_frame, text="12번 Auto Loop:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
        ttk.Label(loop_frame, text="박스 개수:").pack(side=tk.LEFT, padx=5)
        self.auto_boxes_var = tk.IntVar(value=4)
        ttk.Spinbox(loop_frame, from_=1, to=20, textvariable=self.auto_boxes_var, 
                   width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(loop_frame, text="개").pack(side=tk.LEFT, padx=5)
        ttk.Button(loop_frame, text="▶ Start Auto Loop", 
                  command=self.on_auto_loop, width=18).pack(side=tk.LEFT, padx=10)
        
        # 13번 - Quick Start
        quick_frame = ttk.Frame(frame)
        quick_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(quick_frame, text="13번 Quick Start:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
        ttk.Label(quick_frame, text="자동 초기화 + 4개 박스", foreground="blue").pack(side=tk.LEFT, padx=5)
        ttk.Button(quick_frame, text="🚀 Quick Start", 
                  command=self.on_quick_start, width=18).pack(side=tk.LEFT, padx=10)
    
    def create_output_frame(self):
        """출력 프레임"""
        frame = ttk.LabelFrame(self.root, text="Status Output", padding=10)
        frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.output_text = tk.Text(frame, height=20, width=60, font=("Courier", 9))
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.output_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.config(yscrollcommand=scrollbar.set)
    
    # -------------------------
    # 이벤트 핸들러
    # -------------------------
    def on_connect(self):
        """연결"""
        if self.robot_connector.connect():
            self.status_label.config(text="Status: Connected", foreground="green")
            self.log("✓ Robot connected")
        else:
            self.log("✗ Connection failed")
            messagebox.showerror("Error", "Connection failed")
    
    def on_disconnect(self):
        """연결 해제"""
        if self.robot_connector.disconnect():
            self.status_label.config(text="Status: Disconnected", foreground="red")
            self.log("✓ Robot disconnected")
    
    def on_stop(self):
        """정지"""
        self.robot_connector.stop_motion()
        self.log("🛑 STOP")
    
    def on_pause(self):
        """일시정지"""
        self.robot_connector.pause_motion()
        self.log("⏸ PAUSE")
    
    def on_resume(self):
        """재개"""
        self.robot_connector.resume_motion()
        self.log("▶ RESUME")
    
    def on_go_home(self):
        """Home 이동"""
        def home_thread():
            if self.return_home:
                result = self.return_home.cmd_home_only()
                if result.get("ok"):
                    self.log("✓ Moved to home")
                else:
                    self.log("✗ Failed to move home")
            else:
                # return_home이 없으면 직접 이동
                self.robot_motion.move_cart(gui_config.HOME_POSE)
                self.log("✓ Moved to home")
        
        threading.Thread(target=home_thread, daemon=True).start()
    
    def on_joint_move(self, joint_index: int, direction: int):
        """Joint 이동"""
        def joint_thread():
            delta = direction * gui_config.JOINT_INCREMENT
            err = self.robot_motion.rotate_single_joint(joint_index, delta)
            if err == 0:
                self.log(f"✓ J{joint_index+1} moved {delta:+.1f}°")
            else:
                self.log(f"✗ J{joint_index+1} move failed")
        
        threading.Thread(target=joint_thread, daemon=True).start()
    
    def on_gripper_open(self):
        """그리퍼 열기"""
        self.gripper.open()
        self.log("✓ Gripper opened")
    
    def on_gripper_close(self):
        """그리퍼 닫기"""
        self.gripper.close()
        self.log("✓ Gripper closed")
    
    def on_refresh_pose(self):
        """Pose 읽기"""
        (e_p, pose), (e_j, joint) = self.robot_state.read_pose_joint()
        if e_p == 0 and pose:
            pose_str = f"[{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}, {pose[3]:.3f}, {pose[4]:.3f}, {pose[5]:.3f}]"
            self.pose_label.config(text=pose_str)
            self.log(f"Current pose: {pose_str}")
        else:
            self.log("✗ Failed to read pose")
    
    def on_save_home(self):
        """현재 위치를 Home으로 저장"""
        (e_p, pose), (e_j, joint) = self.robot_state.read_pose_joint()
        if e_j == 0 and joint:
            self.robot_state.set_initial_joint6(joint)
            self.log(f"✓ Home position saved: J6={joint[5]:.3f}°")
            messagebox.showinfo("Success", "Home position saved!")
        else:
            self.log("✗ Failed to save home")
            messagebox.showerror("Error", "Could not read current position")
    
    def on_save_pick(self):
        """Pick Pose 저장"""
        self._save_pose("pick_pose", "Pick")
    
    def on_save_safe_pick(self):
        """Safe Pick Pose 저장"""
        self._save_pose("safe_pick", "Safe Pick")
    
    def on_save_place(self):
        """Place Pose 저장"""
        self._save_pose("place_pose", "Place")
    
    def on_save_safe_place(self):
        """Safe Place Pose 저장"""
        self._save_pose("safe_place", "Safe Place")
    
    def _save_pose(self, pose_type: str, label: str):
        """Pose 저장 공통 로직"""
        (e_p, pose), (e_j, joint) = self.robot_state.read_pose_joint()
        if e_p == 0 and pose:
            table = self.table_var.get()
            self.pose_manager.set_table_pose(table, pose_type, pose)
            if self.pose_manager.save():
                self.log(f"✓ {label} pose saved (Table {table})")
                messagebox.showinfo("Success", f"{label} pose saved for Table {table}")
            else:
                self.log(f"✗ Failed to save {label} pose")
        else:
            messagebox.showerror("Error", "Could not get current pose")
    
    def on_view_poses(self):
        """저장된 Pose 보기"""
        table = self.table_var.get()
        poses = self.pose_manager.get_table_poses(table)
        
        self.log("=== Saved Poses ===")
        self.log(f"Table {table}:")
        self.log(f"  Pick: {poses.get('pick_pose')}")
        self.log(f"  Safe Pick: {poses.get('safe_pick')}")
        self.log(f"  Place: {poses.get('place_pose')}")
        self.log(f"  Safe Place: {poses.get('safe_place')}")
    
    def on_save_params(self):
        """파라미터 저장"""
        try:
            self.pose_manager.set_param("num_boxes", self.num_boxes_var.get())
            self.pose_manager.set_param("box_height", self.box_height_var.get())
            self.pose_manager.set_param("vel_normal", self.vel_normal_var.get())
            self.pose_manager.set_param("vel_slow", self.vel_slow_var.get())
            
            if self.pose_manager.save():
                self.log("✓ Parameters saved")
                messagebox.showinfo("Success", "Parameters saved!")
            else:
                self.log("✗ Failed to save parameters")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid parameters: {e}")
    
    def on_start_stacking(self):
        """스택 시작"""
        messagebox.showinfo("Info", "Stacking feature coming soon!")
        self.log("ℹ Stacking feature not implemented yet")
    
    # -------------------------
    # Vision 제어 (NEW!)
    # -------------------------
    def on_vision_on(self):
        """Vision 시작 (2번)"""
        if not self.box_detector:
            self.log("✗ BoxDetector not initialized")
            messagebox.showerror("Error", "Vision module not available")
            return
        
        self.log("[Vision] Starting camera and YOLO...")
        success = self.box_detector.start()
        
        if success:
            self.vision_status_label.config(text="Status: ON", foreground="green")
            self.btn_vision_on.config(state=tk.DISABLED)
            self.btn_vision_off.config(state=tk.NORMAL)
            self.log("✓ Vision started successfully")
            self.log("  - Camera: RealSense D405")
            self.log("  - YOLO: OBB Detection ready")
        else:
            self.log("✗ Vision start failed")
            messagebox.showerror("Error", "Failed to start vision")
    
    def on_vision_off(self):
        """Vision 중지"""
        if not self.box_detector:
            return
        
        self.log("[Vision] Stopping...")
        self.box_detector.stop()
        
        self.vision_status_label.config(text="Status: OFF", foreground="red")
        self.btn_vision_on.config(state=tk.NORMAL)
        self.btn_vision_off.config(state=tk.DISABLED)
        self.log("✓ Vision stopped")
    
    # -------------------------
    # Auto Loop (12번)
    # -------------------------
    def on_auto_loop(self):
        """Custom Auto Loop - N개 박스 이동 (12번)"""
        if not self.auto_pick_place:
            self.log("✗ AutoPickPlace not initialized")
            messagebox.showerror("Error", "Auto pick&place module not available")
            return
        
        num_boxes = self.auto_boxes_var.get()
        
        # 확인 메시지
        answer = messagebox.askyesno(
            "Auto Loop (12번)",
            f"{num_boxes}개 박스를 자동으로 이동합니다.\n\n"
            "• 로봇 연결 필요\n"
            "• Vision 실행 필요\n\n"
            "계속하시겠습니까?"
        )
        
        if not answer:
            return
        
        # 백그라운드 스레드에서 실행
        def auto_loop_thread():
            self.log("=" * 50)
            self.log(f"▶ Auto Loop (12번) 시작! ({num_boxes}개)")
            self.log("=" * 50)
            
            result = self.auto_pick_place.cmd12_auto_loop(num_cycles=num_boxes)
            
            if result.get("ok"):
                counter = result.get("stack_counter", 0)
                self.update_stack_counter(counter)
                self.log("=" * 50)
                self.log(f"✅ Auto Loop 완료! (총 {counter}개 이동)")
                self.log("=" * 50)
                messagebox.showinfo("Success", f"Auto Loop 완료!\n총 {counter}개 박스 이동")
            else:
                msg = result.get("msg", "Unknown error")
                self.log(f"❌ Auto Loop 실패: {msg}")
                messagebox.showerror("Error", f"Auto Loop 실패:\n{msg}")
        
        threading.Thread(target=auto_loop_thread, daemon=True).start()
    
    # -------------------------
    # Quick Start (13번)
    # -------------------------
    def on_quick_start(self):
        """Quick Start - 자동 초기화 + 4개 박스 이동 (13번)"""
        if not self.auto_pick_place:
            self.log("✗ AutoPickPlace not initialized")
            messagebox.showerror("Error", "Auto pick&place module not available")
            return
        
        # 확인 메시지
        answer = messagebox.askyesno(
            "Quick Start (13번)",
            "자동 초기화 후 4개 박스를 이동합니다.\n\n"
            "• 로봇 연결 체크\n"
            "• Vision 시작 체크\n"
            "• Home 저장\n"
            "• 4개 박스 자동 이동\n\n"
            "계속하시겠습니까?"
        )
        
        if not answer:
            return
        
        # 백그라운드 스레드에서 실행
        def quick_start_thread():
            self.log("=" * 50)
            self.log("🚀 Quick Start (13번) 시작!")
            self.log("=" * 50)
            
            result = self.auto_pick_place.cmd13_quick_start()
            
            if result.get("ok"):
                counter = result.get("stack_counter", 0)
                self.update_stack_counter(counter)
                self.log("=" * 50)
                self.log(f"✅ Quick Start 완료! (총 {counter}개 이동)")
                self.log("=" * 50)
                messagebox.showinfo("Success", f"Quick Start 완료!\n총 {counter}개 박스 이동")
            else:
                msg = result.get("msg", "Unknown error")
                self.log(f"❌ Quick Start 실패: {msg}")
                messagebox.showerror("Error", f"Quick Start 실패:\n{msg}")
        
        threading.Thread(target=quick_start_thread, daemon=True).start()
    
    # -------------------------
    # Stack Counter
    # -------------------------
    def update_stack_counter(self, count: int):
        """스택 카운터 업데이트"""
        self.stack_counter_label.config(text=str(count))
        self.log(f"📦 Stack Counter: {count}")
    
    def on_reset_counter(self):
        """스택 카운터 리셋"""
        if not self.auto_pick_place:
            self.log("✗ AutoPickPlace not initialized")
            return
        
        answer = messagebox.askyesno(
            "Reset Counter",
            "스택 카운터를 0으로 리셋하시겠습니까?"
        )
        
        if answer:
            self.auto_pick_place.reset_stack_counter()
            self.update_stack_counter(0)
            self.log("🔄 Stack counter reset to 0")
    
    # -------------------------
    # AMR 제어
    # -------------------------
    def on_amr_goto(self, point_name: str):
        """AMR 이동"""
        if not self.amr or not self.amr_config:
            self.log("✗ AMR not initialized")
            return
        
        def amr_thread():
            try:
                self.log(f"[AMR] Moving to {point_name}...")
                self.amr.set_speed_limit(5.0)
                
                # Table1은 회전 후 이동
                if point_name == "Table1":
                    action = self.amr.go_to(
                        target_x=self.amr_config['points']['Rotate1']['x'],
                        target_y=self.amr_config['points']['Rotate1']['y'],
                        yaw=self.amr_config['points']['Rotate1']['yaw'],
                        speed_ratio=2.0
                    )
                    self.amr._wait_for_action_completion(action, ip=self.amr_config['amr']['Banana'])
                
                # 목표 지점으로 이동
                self.amr.go_to(
                    target_x=self.amr_config['points'][point_name]['x'],
                    target_y=self.amr_config['points'][point_name]['y'],
                    yaw=self.amr_config['points'][point_name]['yaw'],
                    speed_ratio=2.0
                )
                
                # Table1은 거리 체크 후 감속
                if point_name == "Table1":
                    current_position = self.amr.get_robot_pose()
                    while True:
                        distance, _ = compute_distance_and_angle(
                            current_pose=current_position,
                            target_pose=self.amr_config['points'][point_name]
                        )
                        self.log(f"[AMR] Distance: {distance:.2f}m")
                        if distance < 1.0:
                            break
                        current_position = self.amr.get_robot_pose()
                    
                    self.amr.set_speed_limit(0.2)
                
                self.log(f"✓ AMR arrived at {point_name}")
                
                # 로봇 팔 카메라 위치로 이동
                self.log("[AMR] Moving robot arm to camera pose...")
                self.robot_motion.move_cart(gui_config.CAM_POSE)
                self.log("✓ Robot arm ready")
                
            except Exception as e:
                self.log(f"✗ AMR error: {e}")
        
        threading.Thread(target=amr_thread, daemon=True).start()
    
    def on_amr_stop(self):
        """AMR 정지"""
        if self.amr:
            try:
                self.amr.stop_motion_now()
                self.log("🛑 AMR stopped")
            except Exception as e:
                self.log(f"✗ AMR stop failed: {e}")
    
    def _on_mousewheel(self, event):
        """마우스 휠 스크롤"""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def log(self, message: str):
        """로그 출력"""
        self.output_text.insert(tk.END, f"{message}\n")
        self.output_text.see(tk.END)
        self.root.update()