import os, torch, shutil
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs, QuantizationType, QuantizationStrategy

# 1. 환경 및 경로 설정
MODEL_ID = "/home/jinsan/LG_Aimers_2026/00_base_model"
WORKSPACE_DIR = "/home/jinsan/LG_Aimers_2026/03_submission"
OUT_DIR = os.path.join(WORKSPACE_DIR, "model")

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 512

print(f"[INFO] v18(Speed & Accuracy) 실행 중 (B64 + D0.1 + No-Shuffle)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto")

# [데이터 전략] 셔플 제거: 10분 28초를 기록했던 v11.6의 데이터 환경으로 복귀
print(f"[INFO] 데이터셋 로드 중 (Non-Shuffle)...")
ds = load_dataset("LGAI-EXAONE/MANTA-1M", split="train")
ds = ds.select(range(NUM_CALIBRATION_SAMPLES))

def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["conversations"], add_generation_prompt=True, tokenize=False)}

ds = ds.map(preprocess)

# 2. v18 레시피: "10분 벽 돌파를 위한 정밀 세팅"
custom_scheme = QuantizationScheme(
    targets=["Linear"],
    weights=QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.GROUP,
        group_size=128,      
        symmetric=True,
        dynamic=False,
        actorder="static"
    )
)

recipe = [
    GPTQModifier(
        config_groups={"group_0": custom_scheme},
        ignore=["embed_tokens", "lm_head"],
        block_size=64,       # [핵심] 오차 보정 해상도를 높여 지능 사수
        dampening_frac=0.1,  # [핵심] 수치 안정성을 높여 헛소리(Hallucination) 억제
    )
]

print(f"[INFO] v18 양자화 시작 (One-shot)...")
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# 3. 저장 및 압축
if os.path.exists(OUT_DIR): shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

shutil.make_archive(os.path.join(WORKSPACE_DIR, "..", "submit_v18_Final"), "zip", WORKSPACE_DIR, "model")
print(f"✅ v18 생성 완료! 10분 이내의 속도와 0.60 이상의 점수를 기대합니다.")