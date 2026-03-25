import os
import torch
import shutil
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# 1. 경로 및 하이퍼파라미터 설정
MODEL_ID = "/home/jinsan/LG_Aimers_2026/00_base_model"
WORKSPACE_DIR = "/home/jinsan/LG_Aimers_2026/03_submission"
OUT_DIR = os.path.join(WORKSPACE_DIR, "model")
DATASET_ID = "LGAI-EXAONE/MANTA-1M"

# 고정밀 보존을 위한 512 샘플 전략 유지
NUM_CALIBRATION_SAMPLES = 512   
MAX_SEQUENCE_LENGTH = 512       

print(f"[INFO] v11.6 모델 로드 중 (Ultra-Precision GPTQ 전략)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,   # BF16 정밀도 로드 (v5 성공 공식)
    device_map="auto"
)

print("[INFO] 캘리브레이션 데이터 준비 (S512/L512)...")
ds = load_dataset(DATASET_ID, split=f"train[:{NUM_CALIBRATION_SAMPLES}]")

def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["conversations"], add_generation_prompt=True, tokenize=False)}

ds = ds.map(preprocess)

# 2. 초정밀 GPTQ 설정 (block_size=64 & actorder="static")
# block_size를 줄여(128->64) 오차 보정의 해상도를 2배 높입니다.
recipe = [
    GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["embed_tokens", "lm_head"],
        # [핵심] group_size를 64로 설정하여 수치 데이터 복원력 극대화
        block_size=64,          
        # 중요 가중치 정렬 보호
        actorder="static",      
        # 행렬 연산 수치 안정성 확보
        dampening_frac=0.01     
    )
]

print(f"[INFO] 초정밀 GPTQ 양자화 시작 (block_size=64, actorder='static')...")
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# 3. 저장 및 제출용 압축
print("[INFO] v11.6 모델 저장 및 압축 중...")
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

shutil.make_archive(
    base_name=os.path.join(WORKSPACE_DIR, "..", "submit_v11_6_UltraPrecisionGPTQ"), 
    format="zip",
    root_dir=WORKSPACE_DIR,
    base_dir="model"
)
print(f"✅ v11.6(Ultra-Precision) 생성 완료! 이제 0.61 고지를 향해 제출합시다.")