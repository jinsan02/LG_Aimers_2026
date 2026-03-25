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

# --- [v4 핵심 수정: 극한의 정밀도 설정] ---
NUM_CALIBRATION_SAMPLES = 1024   # 256 -> 1024 (데이터 양 4배 증가)
MAX_SEQUENCE_LENGTH = 1024       # 512 -> 1024 (문맥 이해도 2배 증가)
# ---------------------------------------

print("[INFO] 모델 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# [수정] EXAONE 4.0 전용 타겟팅 (정밀 압축을 위해 더 구체적으로 지정)
TARGETS = [
    "re:model.layers.\d+\.self_attn\.(q|k|v|o)_proj",
    "re:model.layers.\d+\.mlp\.(gate|up|down)_proj"
]
IGNORE = ["embed_tokens", "lm_head"]

print("[INFO] 캘리브레이션 데이터 로드 및 스마트 토큰화...")
# KeyError 방지를 위한 안전한 데이터 로딩
ds = load_dataset(DATASET_ID, split="train").shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

def tokenize_fn(sample):
    if "text" in sample: content = sample["text"]
    elif "messages" in sample: content = tokenizer.apply_chat_template(sample["messages"], tokenize=False)
    elif "conversations" in sample: content = tokenizer.apply_chat_template(sample["conversations"], tokenize=False)
    else: content = sample[list(sample.keys())[0]]

    return tokenizer(
        content, 
        padding=False, 
        max_length=MAX_SEQUENCE_LENGTH, 
        truncation=True, 
        add_special_tokens=False
    )

ds = ds.map(tokenize_fn, remove_columns=ds.column_names)

print(f"[INFO] Ultra-Precision GPTQ 시작... (데이터가 많아 40~60분 예상)")
recipe = [GPTQModifier(scheme="W4A16", targets=TARGETS, ignore=IGNORE)]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("[INFO] 모델 저장 및 압축 중...")
os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

shutil.make_archive(
    base_name=os.path.join(WORKSPACE_DIR, "..", "submit"), 
    format="zip",
    root_dir=WORKSPACE_DIR,
    base_dir="model"
)
print(f"✅ v4 생성 완료: ~/LG_Aimers_2026/submit.zip")