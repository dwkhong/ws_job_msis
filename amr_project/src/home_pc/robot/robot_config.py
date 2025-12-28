# robot_config.py

# (선택) 상태 읽기 RPC retry 설정
RPC_RETRY = 1
RPC_RETRY_SLEEP_SEC = 0.25

# IK reference (Fairino SDK에서 보통 0 사용)
IK_REF = 0

# CMD4 search limits
SEARCH_TIMEOUT_SEC = 6.0
SEARCH_MAX_TRIES = 900

# "크게 안 늘리고" 기존 느낌 유지
SEARCH_RZ_LIST = [0, 1, -1, 2, -2, 3, -3, 5, -5, 8, -8, 12, -12, 15, -15, 20, -20, 30, -30]
SEARCH_RX_LIST = [0, 1, -1, 2, -2, 3, -3, 5, -5, 8, -8, 12, -12, 15, -15]
SEARCH_RY_LIST = [0, 1, -1, 2, -2, 3, -3, 5, -5, 8, -8, 12, -12, 15, -15]

# ============================================================
# ✅ SPEED CONFIG (여기서 %로 조절)
# ============================================================
MOVE_CART_VEL_DEFAULT = 100.0
MOVE_CART_VEL_FALLBACKS = [10.0, 5.0, 3.0]     # 실패/112 나오면 순차 적용
MOVE_CART_VEL_LIST = [MOVE_CART_VEL_DEFAULT] + MOVE_CART_VEL_FALLBACKS

MOVEJ_VEL_J6 = 100.0
MOVEJ_BLENDT_J6 = -1.0

MOVEJ_VEL_RETURN = 100.0
MOVEJ_BLENDT_RETURN = -1.0

# ✅ 11번용 MoveJ 속도
MOVEJ_VEL_WP11 = 100.0
MOVEJ_BLENDT_WP11 = -1.0

# -------------------------
# Step config (7번용 유지)
# -------------------------
STEP_SCALE_DEFAULT = 0.3
X_SCALE_MULT = 2.0

# -------------------------
# ✅ 2-Phase approach
# -------------------------
Z_HOLD_OFFSET_MM = 70.0
XY_TOL_MM = 1
Z_TOL_MM  = 2

# -------------------------
# ✅ 6번(J6 회전)
# -------------------------
ANGLE_TO_J6_SIGN = +1.0
J6_MAX_STEP_DEG  = 45.0

# -------------------------
# ✅ 7/9 step try config (7번용 유지)
# -------------------------
STEP_TRY_LIST_DEFAULT = [STEP_SCALE_DEFAULT, 0.05, 0.02, 0.01]

# -------------------------
# ✅ 9번(자동) safety
# -------------------------
AUTO_MAX_SECONDS = 60.0

# ============================================================
# ✅ GRIPPER CONFIG (10번/9번 후 자동닫기)
# ============================================================
GRIPPER_INDEX = 1
GRIPPER_MAX_TIME = 30000
GRIPPER_SPEED = 90
GRIPPER_FORCE = 50
GRIPPER_BLOCK = 1

GRIP_OPEN_POS = 100
GRIP_CLOSE_POS = 60

# ============================================================
# ✅ 11번 스택 사이클 설정
# ============================================================
WP11_A_POSE = [76.752, 218.531, 466.100, -176.747, 11.609, -128.123]
WP11_A_JOINT = [-134.487, -111.775, 75.336, -65.453, -91.956, 83.763]

WP11_DROP_BASE_POSE = [180.145, 312.801, 228.808, -177.820, 1.463, -127.365]

STACK_Z_STEP_MM = 48.0
STACK_Z_MAX_MM = 600.0  # None이면 제한 없음