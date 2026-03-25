import os
import torch
import shutil
import glob
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

# [필수 임포트]
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from compressed_tensors.quantization import (
    QuantizationScheme, 
    QuantizationArgs, 
    QuantizationType, 
    QuantizationStrategy
)

# ==========================================
# 1. 환경 및 경로 설정
# ==========================================
BASE_PATH = "/home/jinsan/LG_Aimers_2026"
MODEL_ID = os.path.join(BASE_PATH, "00_base_model")
WORKSPACE_DIR = os.path.join(BASE_PATH, "03_submission")
OUT_DIR = os.path.join(WORKSPACE_DIR, "model")
DATASET_ROOT = os.path.join(BASE_PATH, "datasets")

def get_jsonl_path(folder_name):
    search_path = os.path.join(DATASET_ROOT, folder_name, "*.jsonl")
    files = glob.glob(search_path)
    if not files:
        raise FileNotFoundError(f"[ERROR] {folder_name} 폴더 내에 .jsonl 파일이 없습니다.")
    return files[0]

# [안정화 설정 완료] 
# 총 510개 (170 * 3) / 길이 512 -> RAM 32GB 환경 최적 수치
NUM_SAMPLES_PER_DS = 170 
MAX_SEQUENCE_LENGTH = 512

print(f"[INFO] v25 (FP8-Safe-Final) 실행 중...")
print(f"[INFO] Target: RAM 32GB / VRAM 8GB | Total Samples: {NUM_SAMPLES_PER_DS * 3}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto"
)

# ==========================================
# 2. 다도메인 데이터셋 로드
# ==========================================
print(f"[INFO] 데이터 로드 및 전처리 시작...")

# (1) MANTA (일반 지시어)
ds_manta = load_dataset("LGAI-EXAONE/MANTA-1M", split="train").shuffle(seed=42).select(range(NUM_SAMPLES_PER_DS))
ds_manta = ds_manta.map(lambda ex: {"text": tokenizer.apply_chat_template(ex["conversations"], add_generation_prompt=True, tokenize=False)}, remove_columns=ds_manta.column_names)

# (2) GSM8K (수학 추론)
gsm8k_file = get_jsonl_path("gsm8k")
ds_gsm = load_dataset("json", data_files=gsm8k_file, split="train").shuffle(seed=42).select(range(NUM_SAMPLES_PER_DS))
ds_gsm = ds_gsm.map(lambda ex: {"text": f"Question: {ex['question']}\nAnswer: {ex['answer']}"}, remove_columns=ds_gsm.column_names)

# (3) KMMLU-Redux (한국어 지식)
kmmlu_file = get_jsonl_path("kmmlu_redux")
ds_kmmlu = load_dataset("json", data_files=kmmlu_file, split="train").shuffle(seed=42).select(range(NUM_SAMPLES_PER_DS))
def proc_kmmlu(ex):
    opts = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(ex['options'])])
    return {"text": f"질문: {ex['question']}\n선택지:\n{opts}\n정답: {ex['solution']}"}
ds_kmmlu = ds_kmmlu.map(proc_kmmlu, remove_columns=ds_kmmlu.column_names)

# 데이터 병합
combined_ds = concatenate_datasets([ds_manta, ds_gsm, ds_kmmlu]).shuffle(seed=42)

# ==========================================
# 3. FP8 (W8A8) 레시피
# ==========================================
fp8_scheme = QuantizationScheme(
    targets=["Linear"],
    weights=QuantizationArgs(num_bits=8, type=QuantizationType.FLOAT, strategy=QuantizationStrategy.TENSOR, symmetric=True, dynamic=False),
    input_activations=QuantizationArgs(num_bits=8, type=QuantizationType.FLOAT, strategy=QuantizationStrategy.TENSOR, symmetric=True, dynamic=True)
)

recipe = [
    GPTQModifier(
        config_groups={"group_0": fp8_scheme},
        ignore=["embed_tokens", "lm_head"],
        targets="Linear",
        dampening_frac=0.1
    )
]

# ==========================================
# 4. 실행 및 저장
# ==========================================
print(f"[INFO] 510개 샘플로 FP8 캘리브레이션 시작...")
oneshot(
    model=model,
    dataset=combined_ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=len(combined_ds),
)

print("[INFO] 양자화 완료. 모델 저장 중...")
os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

# [수정 완료] make_archive 인자 오류 수정 (zip_path -> base_name)
zip_name = os.path.join(BASE_PATH, "submit")
shutil.make_archive(base_name=zip_name, format='zip', root_dir=WORKSPACE_DIR, base_dir="model")

print(f"[INFO] 공정 완료: {zip_name}.zip")