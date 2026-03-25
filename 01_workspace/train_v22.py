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

# [Ultimate 전략]   
# 지능 복구를 위해 L을 1024로 높이고, RAM 안정성을 위해 샘플 수를 조절합니다.
NUM_SAMPLES_PER_DS = 128  # 총 384개 샘플
MAX_SEQUENCE_LENGTH = 768 

print(f"[INFO] v22 Ultimate 실행 중: L={MAX_SEQUENCE_LENGTH} | FP8 (W8A8)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto"
)

# ==========================================
# 2. 다도메인 데이터셋 로드 (지능 복구)
# ==========================================
print(f"[INFO] 데이터 로드 및 전처리 시작...")

# (1) MANTA
ds_manta = load_dataset("LGAI-EXAONE/MANTA-1M", split="train").shuffle(seed=42).select(range(NUM_SAMPLES_PER_DS))
ds_manta = ds_manta.map(lambda ex: {"text": tokenizer.apply_chat_template(ex["conversations"], add_generation_prompt=True, tokenize=False)}, remove_columns=ds_manta.column_names)

# (2) GSM8K
gsm8k_file = get_jsonl_path("gsm8k")
ds_gsm = load_dataset("json", data_files=gsm8k_file, split="train").shuffle(seed=42).select(range(NUM_SAMPLES_PER_DS))
ds_gsm = ds_gsm.map(lambda ex: {"text": f"Question: {ex['question']}\nAnswer: {ex['answer']}"}, remove_columns=ds_gsm.column_names)

# (3) KMMLU-Redux
kmmlu_file = get_jsonl_path("kmmlu_redux")
ds_kmmlu = load_dataset("json", data_files=kmmlu_file, split="train").shuffle(seed=42).select(range(NUM_SAMPLES_PER_DS))
def proc_kmmlu(ex):
    opts = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(ex['options'])])
    return {"text": f"질문: {ex['question']}\n선택지:\n{opts}\n정답: {ex['solution']}"}
ds_kmmlu = ds_kmmlu.map(proc_kmmlu, remove_columns=ds_kmmlu.column_names)

combined_ds = concatenate_datasets([ds_manta, ds_gsm, ds_kmmlu]).shuffle(seed=42)

# ==========================================
# 3. v22 레시피: FP8 (W8A8)
# ==========================================
# 에러를 유발한 kv_cache 인자를 제거하고 표준 FP8 스킴을 유지합니다.
fp8_scheme = QuantizationScheme(
    targets=["Linear"],
    weights=QuantizationArgs(
        num_bits=8, type=QuantizationType.FLOAT, strategy=QuantizationStrategy.TENSOR, symmetric=True, dynamic=False
    ),
    input_activations=QuantizationArgs(
        num_bits=8, type=QuantizationType.FLOAT, strategy=QuantizationStrategy.TENSOR, symmetric=True, dynamic=True
    )
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
print(f"[INFO] {len(combined_ds)}개 샘플로 고정밀 FP8 캘리브레이션 시작...")
oneshot(
    model=model,
    dataset=combined_ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=len(combined_ds),
)

print("[INFO] 양자화 완료. 모델 저장 중...")
os.makedirs(OUT_DIR, exist_ok=True)

# vLLM이 FP8 KV Cache를 인식하도록 설정 주입 (필요 시)
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

# [최종] submit.zip 생성
zip_name = os.path.join(BASE_PATH, "submit")
shutil.make_archive(base_name=zip_name, format='zip', root_dir=WORKSPACE_DIR, base_dir="model")

print(f"[INFO] v22 Ultimate 공정 완료: {zip_name}.zip")