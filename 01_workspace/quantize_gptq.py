import os
import torch
import shutil
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# [수정] 버전 호환성을 위해 아래와 같이 import 경로를 변경합니다.
try:
    from llmcompressor.transformers import oneshot
except ImportError:
    from llmcompressor import oneshot

from llmcompressor.modifiers.quantization import GPTQModifier
# 1. 리눅스(WSL) 경로 설정
MODEL_ID = "/home/jinsan/LG_Aimers_2026/00_base_model"
WORKSPACE_DIR = "/home/jinsan/LG_Aimers_2026/03_submission"
OUT_DIR = os.path.join(WORKSPACE_DIR, "model")

DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

# 양자화 설정 (W4A16: 가중치 4비트)
SCHEME = "W4A16"
TARGETS = ["Linear"]
IGNORE  = ["embed_tokens", "lm_head"]

print("[INFO] 모델 로드 중... (VRAM 관리를 위해 bfloat16 사용)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print("[INFO] 캘리브레이션 데이터 로드 중...")
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")

def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["conversations"],
            add_generation_prompt=True,
            tokenize=False)
    }

ds = ds.map(preprocess)

print(f"[INFO] GPTQ 양자화 시작... (약 10~20분 소요)")
recipe = [GPTQModifier(scheme=SCHEME, targets=TARGETS, ignore=IGNORE)]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("[INFO] 양자화 모델 저장 중...")
os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

# 3. 데이콘 규격에 맞는 submit.zip 생성
print(f"[INFO] 데이콘 제출 규격으로 압축 중...")
shutil.make_archive(
    base_name=os.path.join(WORKSPACE_DIR, "..", "submit"), 
    format="zip",
    root_dir=WORKSPACE_DIR,
    base_dir="model"
)
print(f"✅ 생성 완료: ~/LG_Aimers_2026/submit.zip")