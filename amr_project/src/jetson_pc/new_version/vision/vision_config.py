# vision/vision_config.py
# ============================================================
# ✅ VISION CONFIG (robot_config 스타일: 모듈 상수)
# ============================================================

# -------------------------
# Model
# -------------------------
MODEL_PATH = r"/eddie/model/model_obj/best_obj_20251218.engine"
# MODEL_PATH = r"/eddie/model/model_obj/best_obj_20251218.pt"

CONF_THRES = 0.85
IOU_THRES  = 0.85

# YOLO 입력 해상도
IMGSZ = 640
# -------------------------
# Camera stream (D405)
# -------------------------
WIDTH  = 640
HEIGHT = 320
FPS    = 30

# -------------------------
# Sampling
# -------------------------
AVG_N       = 10
TIMEOUT_SEC = 25.0

# -------------------------
# Depth ROI (meters)
# -------------------------
ROI_MARGIN_PX  = 6.0
MIN_ROI_PIXELS = 120
MAD_THRES_M    = 0.020

DEPTH_MIN_M = 0.15
DEPTH_MAX_M = 3.00

# -------------------------
# Sanity / jump filters
# -------------------------
Z_RANGE_MM = (150.0, 1200.0)
JUMP_XY_MM   = 35.0
JUMP_Z_MM    = 60.0
JUMP_ANG_DEG = 10.0
MAX_CONSEC_SKIPS_RESET = 15

# -------------------------
# Preview / overlay
# -------------------------
SHOW_PREVIEW = False
PREVIEW_WIN_NAME = "OBB + center"

SHOW_OVERLAY = True
OVERLAY_FONT_SCALE = 0.6
OVERLAY_THICKNESS  = 2

PRINT_SELECTED_EACH_ACCEPT = True

