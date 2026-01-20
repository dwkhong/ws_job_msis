# config/vision_config.py
"""
Vision 시스템 설정
"""

# ============================================================
# YOLO Model
# ============================================================
MODEL_PATH = "/ws_job_msis/amr_project/src/jetson_pc/model/model_obj/best_obj_20251218.pt"
# MODEL_PATH = "/home/msis/ws_job_msis/amr_project/src/jetson_pc/model/model_obj/best_obj_20251218.engine"

CONF_THRES = 0.85
IOU_THRES = 0.85
IMGSZ = 640  # YOLO 입력 해상도

# ============================================================
# RealSense Camera (D405)
# ============================================================
WIDTH = 640
HEIGHT = 480
FPS = 30

# ============================================================
# Measurement
# ============================================================
AVG_N = 10          # 평균 계산에 사용할 샘플 수
TIMEOUT_SEC = 25.0  # 측정 타임아웃

# ============================================================
# Depth ROI (meters)
# ============================================================
ROI_MARGIN_PX = 6.0     # ROI 마진 (픽셀)
MIN_ROI_PIXELS = 200    # 최소 ROI 픽셀 수 (120 → 200, 더 엄격)
MAD_THRES_M = 0.015     # MAD 임계값 (미터) (0.020 → 0.015, 더 엄격)

DEPTH_MIN_M = 0.15      # 최소 깊이
DEPTH_MAX_M = 3.00      # 최대 깊이

# ============================================================
# Jump Filter (노이즈 필터링)
# ============================================================
Z_RANGE_MM = (150.0, 1200.0)
JUMP_XY_MM = 35.0       # XY 점프 임계값 (mm)
JUMP_Z_MM = 60.0        # Z 점프 임계값 (mm)
JUMP_ANG_DEG = 10.0     # 각도 점프 임계값 (도)
MAX_CONSEC_SKIPS_RESET = 15

# ============================================================
# Preview / Overlay
# ============================================================
SHOW_PREVIEW = False
PREVIEW_WIN_NAME = "OBB + center"

SHOW_OVERLAY = True
OVERLAY_FONT_SCALE = 0.6
OVERLAY_THICKNESS = 2

PRINT_SELECTED_EACH_ACCEPT = True

# ============================================================
# Stack Counting (Depth-based)
# ============================================================
ENABLE_STACK_COUNTING = False      # 스택 카운팅 활성화

# 다중 테이블 높이 지원
USE_MULTI_BASELINE = True          # 여러 테이블 높이 자동 감지
BASELINE_DEPTH_LOW_MM = 560.0      # 낮은 테이블 깊이 (mm)
BASELINE_DEPTH_HIGH_MM = 750.0     # 높은 테이블 깊이 (mm)
BASELINE_THRESHOLD_MM = 650.0      # 구분 기준 (mm) - 이 값보다 크면 높은 테이블

# 단일 테이블 (USE_MULTI_BASELINE=False 시 사용)
BASELINE_DEPTH_MM = 560.0          # 단일 테이블 깊이 (mm)

BOX_HEIGHT_MM = 58.0               # 박스 1개 높이 (mm)
STACK_COUNT_MAX = 10               # 최대 스택 개수
STACK_DEPTH_TOLERANCE_MM = 10.0    # 깊이 오차 허용 범위 (±mm)

