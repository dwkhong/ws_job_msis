import os
import sys
import gc
import torch
import subprocess
from ultralytics import YOLO

# ==========================================
# 1. 경로 설정 (도커 내부 경로 /eddie 사용)
# ==========================================
model_path = "/eddie/model/model_obj/best_obj_20251218.pt"
onnx_path = model_path.replace(".pt", ".onnx")
engine_path = model_path.replace(".pt", ".engine")

print(f"🚀 모델 경로: {model_path}")

# ==========================================
# 2. ONNX 변환 (이미 있으면 건너뜀)
# ==========================================
if not os.path.exists(onnx_path):
    print("\n[1/2] ONNX 변환 시작...")
    try:
        model = YOLO(model_path)
        # opset=17이 최신 TensorRT와 호환성이 좋습니다
        model.export(format="onnx", opset=17, simplify=True)
        print("✅ ONNX 변환 완료!")
    except Exception as e:
        print(f"❌ ONNX 변환 실패: {e}")
        sys.exit(1)
    
    # 중요: 메모리 확보를 위해 모델 삭제 및 캐시 비우기
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("🧹 메모리 청소 완료 (PyTorch 캐시 삭제됨)")
else:
    print("\n[1/2] ONNX 파일이 이미 존재합니다. 변환 단계를 건너뜁니다.")

# ==========================================
# 3. TensorRT Engine 변환 (trtexec 사용)
# ==========================================
print("\n[2/2] TensorRT Engine 변환 시작 (trtexec)...")

# 파이썬 API 대신 trtexec를 서브프로세스로 실행해야 메모리를 덜 먹습니다.
trtexec_cmd = [
    "/usr/src/tensorrt/bin/trtexec",
    f"--onnx={onnx_path}",
    f"--saveEngine={engine_path}",
    "--fp16"  # 젯슨 성능 최적화 (반정밀도)
]

try:
    print(f"명령어 실행: {' '.join(trtexec_cmd)}")
    subprocess.run(trtexec_cmd, check=True)
    print("\n🎉 모든 변환 작업 성공!")
    print(f"생성된 파일: {engine_path}")
except subprocess.CalledProcessError as e:
    print("\n❌ TensorRT 변환 중 에러 발생 (메모리 부족일 가능성 있음)")
    print("만약 계속 실패하면, 재부팅 후 터미널에서 직접 'trtexec' 명령어를 쓰세요.")