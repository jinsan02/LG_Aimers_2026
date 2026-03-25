import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.pruning.wanda import WandaPruningModifier
from datasets import load_dataset

# 1. 설정 (v3 중도파: 8GB VRAM 최적화)
MODEL_ID = "/home/jinsan/LG_Aimers_2026/00_base_model"
SAVE_DIR = "/home/jinsan/LG_Aimers_2026/03_submission/model"
DATASET_ID = "LGAI-EXAONE/MANTA-1M"

# --- [수정 포인트 1: 하이퍼파라미터] ---
SPARSITY = 0.15           # 25% -> 15%로 완화 (지능 복구)
NUM_SAMPLES = 512         # 256 -> 512개로 증가 (정밀도 보정)
MAX_SEQ_LEN = 512         # 문맥 파악 길이 유지
# -----------------------------------

# 2. 모델 로드
print(f"[INFO] v3 중도파 모델 로드 중 (Sparsity: {SPARSITY})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="auto", trust_remote_code=True
)

# 3. 경량화 레시피 (EXAONE 전용 타겟팅 유지)
TARGETS = [
    "re:model.layers.\d+\.self_attn\.(q|k|v|o)_proj",
    "re:model.layers.\d+\.mlp\.(gate|up|down)_proj"
]

recipe = [
    WandaPruningModifier(sparsity=SPARSITY, targets=TARGETS),
    GPTQModifier(targets=TARGETS, scheme="W4A16", ignore=["embed_tokens", "lm_head"]),
]

# 4. 스마트 토큰화 (KeyError 방어 로직)
def tokenize_fn(sample):
    if "text" in sample: content = sample["text"]
    elif "messages" in sample: content = tokenizer.apply_chat_template(sample["messages"], tokenize=False)
    else: content = sample[list(sample.keys())[0]]

    return tokenizer(
        content, padding=False, max_length=MAX_SEQ_LEN, truncation=True, add_special_tokens=False
    )

print(f"[INFO] 데이터셋 준비 중 (Samples: {NUM_SAMPLES})...")
ds = load_dataset(DATASET_ID, split="train").shuffle(seed=42).select(range(NUM_SAMPLES))
ds = ds.map(tokenize_fn, remove_columns=ds.column_names)

# 5. 하이브리드 압축 실행
print(f"[INFO] v3 중도파 압축 시작...")
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQ_LEN,
    num_calibration_samples=NUM_SAMPLES,
)

# 6. 저장 (vLLM 호환용)
print(f"[INFO] 모델 저장 중: {SAVE_DIR}")
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print(f"✅ v3 하이브리드 모델 생성 완료! (Sparsity: {SPARSITY})")