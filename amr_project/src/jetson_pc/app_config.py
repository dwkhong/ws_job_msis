
# app_config.py
# ============================================================
# ✅ 공용 설정 모음
# ============================================================

# 수정 전 (상대 경로 위험함):
# FAIRINO_PYD_PATH = "../driver/fairino-python-sdk-main/linux/fairino/build/lib.linux-aarch64-3.10"

# 수정 후 (도커 기준 절대 경로로 변경):
FAIRINO_PYD_PATH = "/eddie/driver/fairino-python-sdk-main/linux/fairino/build/lib.linux-aarch64-3.10"

ROBOT_IP_DEFAULT = "192.168.0.15"