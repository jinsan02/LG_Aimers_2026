import os
import torch
import shutil
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# 1. 경로 설정
MODEL_ID = "/home/jinsan/LG_Aimers_2026/00_base_model"
WORKSPACE_DIR = "/home/jinsan/LG_Aimers_2026/03_submission"
OUT_DIR = os.path.join(WORKSPACE_DIR, "model")
DATASET_ID = "LGAI-EXAONE/MANTA-1M"

# --- [v5 핵심 전략: 안정적인 고득점] ---
NUM_CALIBRATION_SAMPLES = 512   # v1(256)보다 2배 늘려 지능 강화
MAX_SEQUENCE_LENGTH = 512       # v1과 동일하게 유지하여 속도 복구
# ------------------------------------

print("[INFO] v5 파이널 모델 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16, # 경고 해결: torch_dtype 대신 dtype 사용
    device_map="auto"
)

print("[INFO] 캘리브레이션 데이터 준비...")
ds = load_dataset(DATASET_ID, split=f"train[:{NUM_CALIBRATION_SAMPLES}]")

def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["conversations"], add_generation_prompt=True, tokenize=False)}

ds = ds.map(preprocess)

# v1에서 성공했던 "Linear" 전체 타겟팅 방식 채택
print(f"[INFO] GPTQ 양자화 시작 (Targets: Linear)...")
recipe = [GPTQModifier(scheme="W4A16", targets=["Linear"], ignore=["embed_tokens", "lm_head"])]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("[INFO] 저장 및 압축 중...")
os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

shutil.make_archive(
    base_name=os.path.join(WORKSPACE_DIR, "..", "submit"), 
    format="zip",
    root_dir=WORKSPACE_DIR,
    base_dir="model"
)
print(f"✅ v5 생성 완료! 이제 이 파일을 제출하세요.")