# demo_gui.py
"""
GUI 데모 - 외형만 보기 (로봇/AMR 연결 없음)
"""
import tkinter as tk
from tkinter import ttk, messagebox


class DemoGUI:
    """GUI 데모"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Fairino Robot Control - DEMO")
        self.root.geometry("900x800")
        
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
        
        # 1. Connection
        self.create_connection_frame()
        
        # 2. Controls
        self.create_control_frame()
        
        # 3. Vision Control (NEW!)
        self.create_vision_frame()
        
        # 4. Joint Movement
        self.create_joint_frame()
        
        # 4. Gripper
        self.create_gripper_frame()
        
        # 5. Current Pose
        self.create_pose_frame()
        
        # 6. Pose Management
        self.create_save_frame()
        
        # 7. Stacking Parameters
        self.create_param_frame()
        
        # 8. Stacking Control (Original)
        self.create_run_frame()
        
        # 9. Auto Control (12번, 13번)
        self.create_auto_control_frame()
        
        # 우측 프레임 (Output)
        self.create_output_frame()
    
    def _on_mousewheel(self, event):
        """마우스 휠 스크롤"""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def create_connection_frame(self):
        """연결 프레임"""
        frame = ttk.LabelFrame(self.left_frame, text="Robot Connection", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(frame, text="Robot IP:").grid(row=0, column=0, sticky=tk.W)
        
        self.ip_var = tk.StringVar(value="192.168.0.15")
        ttk.Entry(frame, textvariable=self.ip_var, width=20).grid(row=0, column=1, sticky=tk.W, padx=5)
        
        ttk.Button(frame, text="Connect", command=lambda: self.log("Connect clicked (demo)")).grid(row=0, column=2, padx=5)
        ttk.Button(frame, text="Disconnect", command=lambda: self.log("Disconnect clicked (demo)")).grid(row=0, column=3, padx=5)
        
        self.status_label = ttk.Label(frame, text="Status: Disconnected (DEMO MODE)", foreground="orange")
        self.status_label.grid(row=0, column=4, padx=10)
    
    def create_control_frame(self):
        """제어 프레임"""
        frame = ttk.LabelFrame(self.left_frame, text="Robot Controls", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(frame, text="STOP", command=lambda: self.log("🛑 STOP"), width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="PAUSE", command=lambda: self.log("⏸ PAUSE"), width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="RESUME", command=lambda: self.log("▶ RESUME"), width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="GO HOME", command=lambda: self.log("🏠 GO HOME"), width=15).pack(side=tk.LEFT, padx=5)
        
        # AMR 버튼
        ttk.Button(frame, text="AMR table1", command=lambda: self.log("AMR → Table1"), width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="AMR table2", command=lambda: self.log("AMR → Table2"), width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="AMR stop", command=lambda: self.log("AMR STOP"), width=10).pack(side=tk.LEFT, padx=5)
    
    def create_vision_frame(self):
        """Vision 제어 프레임 (NEW!)"""
        frame = ttk.LabelFrame(self.left_frame, text="Vision Control (Camera + YOLO)", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Vision On/Off 버튼
        self.btn_vision_on = ttk.Button(frame, text="📷 Vision ON", 
                                        command=lambda: self.toggle_vision(True), width=15)
        self.btn_vision_on.pack(side=tk.LEFT, padx=5)
        
        self.btn_vision_off = ttk.Button(frame, text="📷 Vision OFF", 
                                         command=lambda: self.toggle_vision(False), 
                                         width=15, state=tk.DISABLED)
        self.btn_vision_off.pack(side=tk.LEFT, padx=5)
        
        # Vision 상태 표시
        self.vision_status_label = ttk.Label(frame, text="Status: OFF", foreground="red", font=("Arial", 10, "bold"))
        self.vision_status_label.pack(side=tk.LEFT, padx=10)
    
    def toggle_vision(self, is_on):
        """Vision ON/OFF 토글"""
        if is_on:
            self.vision_status_label.config(text="Status: ON", foreground="green")
            self.btn_vision_on.config(state=tk.DISABLED)
            self.btn_vision_off.config(state=tk.NORMAL)
            self.log("✓ Vision started")
            self.log("  - Camera: RealSense D405")
            self.log("  - YOLO: OBB Detection ready")
        else:
            self.vision_status_label.config(text="Status: OFF", foreground="red")
            self.btn_vision_on.config(state=tk.NORMAL)
            self.btn_vision_off.config(state=tk.DISABLED)
            self.log("✓ Vision stopped")
    
    def create_joint_frame(self):
        """Joint 이동 프레임"""
        frame = ttk.LabelFrame(self.left_frame, text="Joint Movement (±5°)", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        for i in range(6):
            sub_frame = ttk.Frame(frame)
            sub_frame.pack(fill=tk.X, padx=5, pady=3)
            
            ttk.Label(sub_frame, text=f"Joint {i+1}:", width=10).pack(side=tk.LEFT)
            ttk.Button(sub_frame, text="←", width=3, 
                      command=lambda j=i: self.log(f"J{j+1} ← -5°")).pack(side=tk.LEFT, padx=2)
            ttk.Button(sub_frame, text="→", width=3, 
                      command=lambda j=i: self.log(f"J{j+1} → +5°")).pack(side=tk.LEFT, padx=2)
    
    def create_gripper_frame(self):
        """그리퍼 프레임"""
        frame = ttk.LabelFrame(self.left_frame, text="Gripper Control", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(frame, text="Open Gripper", command=lambda: self.log("✋ Gripper OPEN"), width=20).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="Close Gripper", command=lambda: self.log("✊ Gripper CLOSE"), width=20).pack(side=tk.LEFT, padx=5)
    
    def create_pose_frame(self):
        """현재 Pose 프레임"""
        frame = ttk.LabelFrame(self.left_frame, text="Current Pose (Cartesian)", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.pose_label = ttk.Label(frame, text="[DEMO] Not available", font=("Courier", 9))
        self.pose_label.pack()
        
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(pady=5)
        
        ttk.Button(btn_frame, text="Refresh Pose", command=lambda: self.log("Refresh Pose (demo)")).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="💾 Save as Home", command=lambda: self.log("✓ Home position saved")).pack(side=tk.LEFT, padx=3)
    
    def create_save_frame(self):
        """Pose 저장 프레임"""
        frame = ttk.LabelFrame(self.left_frame, text="Pose Management", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(frame, text="Save Pick Pose", command=lambda: self.log("💾 Pick Pose saved"), width=18).pack(side=tk.LEFT, padx=3)
        ttk.Button(frame, text="Save Safe Pick", command=lambda: self.log("💾 Safe Pick saved"), width=18).pack(side=tk.LEFT, padx=3)
        ttk.Button(frame, text="Save Place Pose", command=lambda: self.log("💾 Place Pose saved"), width=18).pack(side=tk.LEFT, padx=3)
        ttk.Button(frame, text="Save Safe Place", command=lambda: self.log("💾 Safe Place saved"), width=18).pack(side=tk.LEFT, padx=3)
        ttk.Button(frame, text="View Saved Poses", command=lambda: self.log("📋 View Poses"), width=18).pack(side=tk.LEFT, padx=3)
    
    def create_param_frame(self):
        """스택 파라미터 프레임"""
        frame = ttk.LabelFrame(self.left_frame, text="Stacking Parameters", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Row 1
        ttk.Label(frame, text="Number of Boxes:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.num_boxes_var = tk.IntVar(value=4)
        ttk.Entry(frame, textvariable=self.num_boxes_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(frame, text="Box Height (mm):").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.box_height_var = tk.DoubleVar(value=58.0)
        ttk.Entry(frame, textvariable=self.box_height_var, width=10).grid(row=0, column=3, sticky=tk.W, padx=5)
        
        ttk.Label(frame, text="Table:").grid(row=0, column=4, sticky=tk.W, padx=5)
        self.table_var = tk.StringVar(value="1")
        self.table_combo = ttk.Combobox(frame, textvariable=self.table_var, 
                                        values=["1","2","3","4","5"], width=5, state="readonly")
        self.table_combo.grid(row=0, column=5, sticky=tk.W, padx=5)
        
        # Row 2
        ttk.Label(frame, text="Normal Speed:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.vel_normal_var = tk.DoubleVar(value=30.0)
        ttk.Entry(frame, textvariable=self.vel_normal_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(frame, text="Slow Speed:").grid(row=1, column=2, sticky=tk.W, padx=5)
        self.vel_slow_var = tk.DoubleVar(value=30.0)
        ttk.Entry(frame, textvariable=self.vel_slow_var, width=10).grid(row=1, column=3, sticky=tk.W, padx=5)
        
        ttk.Button(frame, text="Save Parameters", command=lambda: self.log("✓ Parameters saved")).grid(row=1, column=4, columnspan=2, sticky=tk.W, padx=5)
    
    def create_run_frame(self):
        """스택 실행 프레임 (Original)"""
        frame = ttk.LabelFrame(self.left_frame, text="Stacking Control (Original)", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(frame, text="Start Stacking (Manual)", 
                  command=lambda: self.log("🚀 Start Stacking (original - demo)"), 
                  width=25).pack(side=tk.LEFT, padx=5)
    
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
                  command=lambda: self.reset_counter(), width=15).pack(side=tk.LEFT, padx=15)
        
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
    
    def reset_counter(self):
        """카운터 리셋 데모"""
        self.stack_counter_label.config(text="0")
        self.log("🔄 Stack counter reset to 0")
    
    def on_quick_start(self):
        """Quick Start 데모 (13번)"""
        self.log("=" * 50)
        self.log("🚀 Quick Start (13번) - DEMO")
        self.log("=" * 50)
        self.log("1. 로봇 연결 체크...")
        self.log("2. Vision 시작 체크...")
        self.log("3. Home 저장...")
        self.log("4. 스택 카운터 리셋...")
        self.log("5. 4개 박스 자동 이동 중...")
        self.log("")
        for i in range(4):
            self.log(f"   [Box {i+1}/4] Pick → Place")
        self.log("")
        self.log("✅ Quick Start 완료! (총 4개 이동)")
        self.log("=" * 50)
        self.stack_counter_label.config(text="4")
    
    def on_auto_loop(self):
        """Auto Loop 데모 (12번)"""
        num = self.auto_boxes_var.get()
        self.log("=" * 50)
        self.log(f"▶ Auto Loop (12번) - DEMO ({num}개)")
        self.log("=" * 50)
        self.log("1. 로봇 연결 체크...")
        self.log("2. Vision 체크...")
        for i in range(num):
            self.log(f"   [Box {i+1}/{num}] Pick → Place")
        self.log("")
        self.log(f"✅ Auto Loop 완료! (총 {num}개 이동)")
        self.log("=" * 50)
        self.stack_counter_label.config(text=str(num))
    
    def create_output_frame(self):
        """출력 프레임"""
        frame = ttk.LabelFrame(self.root, text="Status Output", padding=10)
        frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.output_text = tk.Text(frame, height=20, width=60, font=("Courier", 9))
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.output_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.config(yscrollcommand=scrollbar.set)
        
        # 초기 메시지
        self.log("=" * 50)
        self.log("GUI DEMO MODE - No robot connection")
        self.log("=" * 50)
        self.log("✨ NEW FEATURES:")
        self.log("  • Vision Control (Camera + YOLO)")
        self.log("  • Quick Start (13번 - Auto 4 boxes)")
        self.log("")
        self.log("Click any button to test GUI layout")
        self.log("")
    
    def log(self, message: str):
        """로그 출력"""
        self.output_text.insert(tk.END, f"{message}\n")
        self.output_text.see(tk.END)
        self.root.update()


if __name__ == "__main__":
    root = tk.Tk()
    app = DemoGUI(root)
    root.mainloop()