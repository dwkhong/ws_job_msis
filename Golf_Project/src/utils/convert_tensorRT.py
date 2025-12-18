# tensorrt_converter.py
# -*- coding: utf-8 -*-

from ultralytics import YOLO
from pathlib import Path

def convert_all_to_trt(
    runs_dir: str | Path,
    img_size: int = 640,
    batch: int = 1,
    fp16: bool = True,
    dynamic: bool = False,
    workspace: int = 4096,
):

    runs_dir = Path(runs_dir)
    pt_list = sorted(runs_dir.rglob("weights/best.pt"))
    print(f"[INFO] best.pt 발견: {len(pt_list)}개")

    precision_tag = "fp16" if fp16 else "fp32"
    dynamic_tag   = "dynamic" if dynamic else "static"

    for idx, pt_path in enumerate(pt_list, start=1):
        print("\n" + "=" * 60)
        print(f"[{idx}/{len(pt_list)}] 처리 중: {pt_path}")
        print("=" * 60)

        engine_name = (
            f"{pt_path.stem}_{precision_tag}_bs{batch}_{img_size}px_{dynamic_tag}.engine"
        )
        engine_path = pt_path.parent / engine_name

        # 이미 엔진 존재하면 스킵
        if engine_path.exists():
            print(f"  ⏭️  SKIP — 이미 존재: {engine_path.name}")
            continue

        # 엔진 생성
        try:
            print("  🔧 엔진 생성 시작...")

            exported_engine = YOLO(str(pt_path)).export(
                format="engine",
                imgsz=img_size,
                device=0,
                half=fp16,
                dynamic=dynamic,
                batch=batch,
                workspace=workspace,
                simplify=True,
                name=engine_path.stem,
            )

            exported_engine = Path(exported_engine)

            if exported_engine.resolve() != engine_path.resolve():
                exported_engine.rename(engine_path)

            print(f"  ✅ 저장 완료: {engine_path}")

        except Exception as e:
            print(f"  ❌ 오류 발생: {e}")
            continue

    print("\n=== 🔥 전체 TensorRT 엔진 변환 완료 ===")
