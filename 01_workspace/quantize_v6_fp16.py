import os
import torch
import shutil
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.pruning.wanda import WandaPruningModifier

# 1. 경로 설정 (기존 v6와 겹치지 않게 분리)
MODEL_ID = "/home/jinsan/LG_Aimers_2026/00_base_model"
WORKSPACE_DIR = "/home/jinsan/LG_Aimers_2026/03_submission"
OUT_DIR = os.path.join(WORKSPACE_DIR, "v6_fp16/model") # 폴더 분리
DATASET_ID = "LGAI-EXAONE/MANTA-1M"

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 512

print("[INFO] v6_fp16: FP16 정밀도 모드로 로또 티켓 사냥 시작...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# 핵심 수정: dtype을 float16으로 변경
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16, # BF16 -> FP16으로 정밀도 업그레이드
    device_map="auto"
)

# 2. 데이터셋 준비 (v5/v6 성공 공식 유지)
ds = load_dataset(DATASET_ID, split=f"train[:{NUM_CALIBRATION_SAMPLES}]")
def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["conversations"], add_generation_prompt=True, tokenize=False)}
ds = ds.map(preprocess)

# 3. v6 반복적 프루닝 레시피
TARGETS = ["Linear"]
recipe = [
    WandaPruningModifier(sparsity=0.10, targets=TARGETS),
    WandaPruningModifier(sparsity=0.20, targets=TARGETS),
    GPTQModifier(targets=TARGETS, scheme="W4A16", ignore=["embed_tokens", "lm_head"])
]

# 4. 압축 실행
print(f"[INFO] FP16 기반 3단계 하이브리드 압축 실행...")
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# 5. 저장 및 제출 파일 생성
print("[INFO] v6_fp16 모델 저장 중...")
os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

# 상위 폴더에 submit_fp16.zip으로 생성하여 기존 파일과 구분
shutil.make_archive(
    base_name=os.path.join(WORKSPACE_DIR, "..", "submit_fp16"), 
    format="zip",
    root_dir=os.path.join(WORKSPACE_DIR, "v6_fp16"),
    base_dir="model"
)
print(f"✅ v6_fp16 생성 완료! ~/LG_Aimers_2026/submit_fp16.zip")