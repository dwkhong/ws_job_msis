from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class VisionConfig:
    # model
    model_path: str
    conf_thres: float = 0.85
    iou_thres: float = 0.75
    imgsz: int = 640

    # camera stream
    width: int = 640
    height: int = 480
    fps: int = 30

    # box real size (mm)
    box_w_mm: float = 230.0
    box_h_mm: float = 95.0

    # sampling
    avg_n: int = 10
    timeout_sec: float = 25.0

    # depth ROI (meters)
    roi_margin_px: float = 6.0
    min_roi_pixels: int = 120
    mad_thres_m: float = 0.020
    depth_min_m: float = 0.15
    depth_max_m: float = 3.00

    # sanity filters
    z_range_mm: Tuple[float, float] = (150.0, 1200.0)
    size_rel_err_max: float = 0.25

    jump_xy_mm: float = 35.0
    jump_z_mm: float = 60.0
    jump_ang_deg: float = 10.0

    max_consec_skips_reset: int = 15

    # preview / overlay
    show_preview: bool = True
    preview_win_name: str = "OBB + center"
    show_overlay: bool = True
    overlay_font_scale: float = 0.6
    overlay_thickness: int = 2
    print_selected_each_accept: bool = True


DEFAULT_VISION_CONFIG = VisionConfig(
    model_path=r"C:\Users\rhdeh\ws_job_msis\amr_project\src\labtop_pc\model\model_obj\best_obj_20251218.pt"
)
