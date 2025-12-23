from pathlib import Path
import sys

# ✅ 현재 파일: .../src/scripts/run_check_matching.py
# parents[1] => .../src
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset.check_label_img_matching_1 import check_label_matching
from scripts.dataset.settings import BASE_DIR  

jpg_only, txt_only = check_label_matching(BASE_DIR)

print(f"jpg_only: {len(jpg_only)}")
print(f"txt_only: {len(txt_only)}")