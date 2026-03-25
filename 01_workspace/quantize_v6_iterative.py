import os
import torch
import shutil
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.pruning.wanda import WandaPruningModifier

# 1. 경로 및 설정 (v5 성공 공식 계승)
MODEL_ID = "/home/jinsan/LG_Aimers_2026/00_base_model"
WORKSPACE_DIR = "/home/jinsan/LG_Aimers_2026/03_submission"
OUT_DIR = os.path.join(WORKSPACE_DIR, "model")
DATASET_ID = "LGAI-EXAONE/MANTA-1M"

NUM_CALIBRATION_SAMPLES = 512   # v5의 승리 공식 유지
MAX_SEQUENCE_LENGTH = 512       # 속도와 VRAM의 균형점

print("[INFO] v6: 로또 티켓 사냥 시작 (Iterative Pruning)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto"
)

# 2. 데이터셋 준비
print("[INFO] 캘리브레이션 데이터 로드 중...")
ds = load_dataset(DATASET_ID, split=f"train[:{NUM_CALIBRATION_SAMPLES}]")

def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["conversations"], add_generation_prompt=True, tokenize=False)}

ds = ds.map(preprocess)

# 3. v6 핵심: 반복적 프루닝 + GPTQ 레시피
# 전체 Linear 레이어를 대상으로 하되, 단계를 나누어 '우승 티켓'을 찾습니다.
TARGETS = ["Linear"]

recipe = [
    # [Step 1] 10% 프루닝: 가장 중요도가 낮은 가중치 먼저 제거
    WandaPruningModifier(sparsity=0.10, targets=TARGETS),
    
    # [Step 2] 20% 프루닝: 모델이 적응한 상태에서 추가 제거
    WandaPruningModifier(sparsity=0.20, targets=TARGETS),
    
    # [Step 3] 최종 양자화: 남은 핵심 뉴런들을 4비트로 압축
    GPTQModifier(targets=TARGETS, scheme="W4A16", ignore=["embed_tokens", "lm_head"])
]

# 4. 압축 실행 (Iterative 과정으로 인해 이전보다 1.5배 정도 시간이 더 걸립니다.)
print(f"[INFO] 3단계 하이브리드 압축 실행...")
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# 5. 저장 및 제출 파일 생성
print("[INFO] v6 모델 저장 및 압축 중...")
os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

shutil.make_archive(
    base_name=os.path.join(WORKSPACE_DIR, "..", "submit"), 
    format="zip",
    root_dir=WORKSPACE_DIR,
    base_dir="model"
)
print(f"✅ v6 생성 완료! ~/LG_Aimers_2026/submit.zip")