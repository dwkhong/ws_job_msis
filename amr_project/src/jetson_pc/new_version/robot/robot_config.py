# robot_config.py

# robot_state.py에서 사용
# 실패시에 얼마나 더 사용할건지 
RPC_RETRY = 1
RPC_RETRY_SLEEP_SEC = 0.25

# -----------------------------
# ✅ Camera -> Gripper Offset (mm)
# -----------------------------
OFF_X_MM = -18.0
OFF_Y_MM = -70.0
OFF_Z_MM = -175.0

PIVOT_LENGTH = 175.0

# ============================================================
# ✅ 로봇 베이스가 "왼쪽으로 약 135도" 회전한 상태 보정
# ============================================================
BASE_YAW_OFFSET_DEG = 90

# ============================================================
# ✅ 축 부호가 뒤집히는 케이스 대응 (딱 여기서만 뒤집기)
# ============================================================
FLIP_MOVE_X = False
FLIP_MOVE_Y = False

# -------------------------
# Tool / User coordinate IDs (MoveCart 등에 사용)
# -------------------------
TOOL_ID = 0
USER_ID = 0



# IK reference (Fairino SDK에서 보통 0 사용)
# ik_adjust.py에서 사용
IK_REF = 0

SEARCH_TIMEOUT_SEC = 6.0
SEARCH_MAX_TRIES = 900

SEARCH_RZ_LIST = [0, 1, -1, 2, -2, 3, -3, 5, -5, 8, -8, 12, -12, 15, -15, 20, -20, 30, -30]
SEARCH_RX_LIST = [0, 1, -1, 2, -2, 3, -3, 5, -5, 8, -8, 12, -12, 15, -15]
SEARCH_RY_LIST = [0, 1, -1, 2, -2, 3, -3, 5, -5, 8, -8, 12, -12, 15, -15]

# j6_rotate.py에서 사용

ANGLE_TO_J6_SIGN = +1.0
J6_MAX_STEP_DEG  = 45.0

MOVEJ_VEL_J6 = 90.0
MOVEJ_BLENDT_J6 = -1.0

# -------------------------
# Step config (7번용 유지)
# -------------------------
STEP_SCALE_DEFAULT = 0.9
X_SCALE_MULT = 2.0

# -------------------------
# ✅ 2-Phase approach
# -------------------------
Z_HOLD_OFFSET_MM = 100.0
XY_TOL_MM = 1
Z_TOL_MM  = 2

# -------------------------
# ✅ 7/9 step try config (7번용 유지)
# -------------------------
STEP_TRY_LIST_DEFAULT = [STEP_SCALE_DEFAULT, 0.05, 0.02, 0.01]

# ============================================================
# ✅ SPEED CONFIG (여기서 %로 조절)
# ============================================================
MOVE_CART_VEL_DEFAULT = 90.0          # ✅ 안전 시작 (100 -> 30)
MOVE_CART_VEL_FALLBACKS = [15.0, 8.0, 3.0]
MOVE_CART_VEL_LIST = [MOVE_CART_VEL_DEFAULT] + MOVE_CART_VEL_FALLBACKS

MOVE_CART_ACC = 50.0                  # ✅ 가속도 낮게
#MOVE_CART_DEC = 80.0                  # (지금 move_step.py에선 안 쓰지만 같이 둠)

MOVE_CART_OVL = 50.0                  # ✅ override(%) 안전 시작 (20%)
MOVE_CART_BLENDT = -1.0               # ✅ blending time (보통 -1 = off/기본)
MOVE_CART_EX = -1                     # ✅ 확장옵션/모드 (기본 0)

MOVEJ_VEL_RETURN = 90.0
MOVEJ_BLENDT_RETURN = -1.0

# ✅ 11번용 MoveJ 속도
MOVEJ_VEL_WP11 = 90.0
MOVEJ_BLENDT_WP11 = -1.0

# -------------------------
# ✅ 9번(자동) safety
# -------------------------

# ============================================================
# ✅ GRIPPER CONFIG (10번/9번 후 자동닫기)
# ============================================================
GRIPPER_INDEX = 1
GRIPPER_MAX_TIME = 30000
GRIPPER_SPEED = 90
GRIPPER_FORCE = 50
GRIPPER_BLOCK = 1

GRIP_OPEN_POS = 100
GRIP_CLOSE_POS = 25

# ============================================================
# ✅ 11번 스택 사이클 설정
# ============================================================
# WP11_A_POSE = [76.752, 218.531, 466.100, -176.747, 11.609, -128.123]
# WP11_A_JOINT = [-134.487, -111.775, 75.336, -65.453, -91.956, 83.763]

# WP11_DROP_BASE_POSE = [180.145, 312.801, 228.808, -177.820, 1.463, -127.365]

WP11_A_POSE = [56.655, -324.827, 515.554, 179.807, 0.914, 89.979]
WP11_A_JOINT = [-62.004, -65.272, -103.647, -100.183, 89.741, 118.013]

WP11_DROP_BASE_POSE = [94.070, -407.414, 247.161, -178.769, -0.710, 90.880]

STACK_Z_STEP_MM = 58.0
STACK_Z_MAX_MM = 600.0  # None이면 제한 없음