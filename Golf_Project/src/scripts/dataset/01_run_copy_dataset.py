from dataset.copy_dataset import copy_folders

SRC_LIST = [
    "/home/dw/ws_job_msislab/Golf_Project/data/20250721/20250721_good_data",
    "/home/dw/ws_job_msislab/Golf_Project/data/20250725/20250725_good_data",
    "/home/dw/ws_job_msislab/Golf_Project/data/20250904/20250904_good_data",
]

DST = "/home/dw/ws_job_msislab/Golf_Project/data/for_study/20251107_merge_data"

copy_folders(SRC_LIST, DST)