import os
import torch
import shutil
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# 1. 경로 및 하이퍼파라미터 설정 (v5 기반 강화)
MODEL_ID = "/home/jinsan/LG_Aimers_2026/00_base_model"
WORKSPACE_DIR = "/home/jinsan/LG_Aimers_2026/03_submission"
OUT_DIR = os.path.join(WORKSPACE_DIR, "model")
DATASET_ID = "LGAI-EXAONE/MANTA-1M"

# --- [v10 수정 전략: 샘플 수 확대로 지능 강화] ---
NUM_CALIBRATION_SAMPLES = 1024  # v5(512)보다 2배 늘려 양자화 정밀도 향상
MAX_SEQUENCE_LENGTH = 512      # v5와 동일하게 유지
# --------------------------------------------

print("[INFO] v10 모델 로드 중 (BF16 & Calibration 강화 전략)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,        # v5의 성공 공식인 BF16 유지
    device_map="auto"
)

print("[INFO] 캘리브레이션 데이터 준비...")
ds = load_dataset(DATASET_ID, split=f"train[:{NUM_CALIBRATION_SAMPLES}]")

def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["conversations"], add_generation_prompt=True, tokenize=False)}

ds = ds.map(preprocess)

# 2. GPTQ 양자화 설정 (오류 수정 완료)
# scheme을 v5에서 성공했던 문자열 프리셋으로 복구합니다.
print(f"[INFO] GPTQ 양자화 시작 (Samples: {NUM_CALIBRATION_SAMPLES})...")
recipe = [
    GPTQModifier(
        targets=["Linear"],      # v5와 동일한 리스트 형식
        ignore=["embed_tokens", "lm_head"],
        scheme="W4A16"           # 검증된 프리셋 사용
    )
]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# 3. 저장 및 제출용 압축
print("[INFO] v10 모델 저장 및 압축 중...")
os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

shutil.make_archive(
    base_name=os.path.join(WORKSPACE_DIR, "..", "submit_v10"), 
    format="zip",
    root_dir=WORKSPACE_DIR,
    base_dir="model"
)
print(f"✅ v10 생성 완료! (Sample 1024 강화 버전)")