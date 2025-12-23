# src/scripts/00_settings.py
from pathlib import Path

# ✅ 복사할 원본 폴더들
SRC_LIST = [
    Path("/home/dw/ws_job_msislab/Golf_Project/data/check_1"),
    Path("/home/dw/ws_job_msislab/Golf_Project/data/check_2"),
]

# ✅ 복사 목적지 루트
DST_ROOT = Path("/home/dw/ws_job_msislab/Golf_Project/data/for_study/20251223_check")

# ✅ 매칭 체크할 베이스 디렉토리
BASE_DIR = DST_ROOT
