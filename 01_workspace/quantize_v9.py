import os
import torch
import shutil
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# --- [1. 경로 및 설정] ---
BASE_PATH = "/home/jinsan/LG_Aimers_2026"
MODEL_ID = f"{BASE_PATH}/00_base_model"
OUT_DIR = f"{BASE_PATH}/03_submission/model"
DATASET_ID = "LGAI-EXAONE/MANTA-1M"

# [V9-Stable 핵심 설정]
# 8GB VRAM(RTX 5060)에서 'Killed' 없이 완주 가능한 안전 수치입니다.
NUM_CALIB_SAMPLES = 512 
MAX_SEQ_LEN = 512

if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# --- [2. 모델 및 데이터 로드] ---
# 공식 지원 환경이므로 원본 토크나이저를 그대로 사용합니다.
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    dtype=torch.float16, # torch_dtype 경고 해결
    device_map="auto", 
    trust_remote_code=True
)

def tokenize_fn(examples):
    full_text = [tokenizer.apply_chat_template(conv, tokenize=False) for conv in examples["conversations"]]
    return tokenizer(full_text, truncation=True, max_length=MAX_SEQ_LEN, padding="max_length")

print(f"🧹 캘리브레이션 데이터 준비 중... (Samples: {NUM_CALIB_SAMPLES})")
ds = load_dataset(DATASET_ID, split="train")
calib_ds = ds.select(range(NUM_CALIB_SAMPLES)).map(
    tokenize_fn, 
    batched=True, 
    remove_columns=ds.column_names 
)

# --- [3. FP8 양자화 적용 (지능 보존형)] ---
print("💎 v9-Stable: FP8(W8A8) Quantization 시작...")
torch.cuda.empty_cache() # 메모리 파편화 방지

oneshot(
    model=model, 
    dataset=calib_ds,
    num_calibration_samples=NUM_CALIB_SAMPLES,
    recipe=[
        QuantizationModifier(
            targets=["Linear"], 
            scheme="FP8", # L4 GPU 하드웨어 가속 규격
            ignore=["embed_tokens", "lm_head"] # 핵심 레이어 보호
        )
    ]
)

# --- [4. 저장 및 아카이브 (서버 규정 준수)] ---
print("💾 모델 저장 중 (Compressed Format)...")

# 대회 서버에 compressed-tensors 라이브러리가 있으므로 
# save_compressed=True로 저장해야 용량도 줄이고 서버 가속도 터집니다.
model.save_pretrained(OUT_DIR, save_compressed=True) 
tokenizer.save_pretrained(OUT_DIR)

# 최종 ZIP 생성
archive_name = f"{BASE_PATH}/V9_Final_Stable_FP8"
shutil.make_archive(
    base_name=archive_name, 
    format="zip",
    root_dir=f"{BASE_PATH}/03_submission",
    base_dir="model"
)

print(f"✅ v9-Stable 생성 완료: {archive_name}.zip")