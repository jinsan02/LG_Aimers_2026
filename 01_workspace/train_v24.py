import os
import torch
import shutil
import glob
import logging
import gc
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

# [1. 필수 임포트 및 모듈 설정]
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from compressed_tensors.quantization import (
    QuantizationScheme, QuantizationArgs, QuantizationType, QuantizationStrategy
)

# ==========================================
# 1. 환경 및 경로 설정
# ==========================================
BASE_PATH = "/home/jinsan/LG_Aimers_2026"
MODEL_ID = os.path.join(BASE_PATH, "00_base_model")
WORKSPACE_DIR = os.path.join(BASE_PATH, "03_submission")
OUT_DIR = os.path.join(WORKSPACE_DIR, "model")
DATASET_ROOT = os.path.join(BASE_PATH, "datasets")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EXAONE_v26_Pure_v2")

# [설정] 샘플 수 128 유지, 시퀀스 길이 768
NUM_SAMPLES_PER_DS = 128 
MAX_SEQUENCE_LENGTH = 768

def get_jsonl_path(folder_name):
    search_path = os.path.join(DATASET_ROOT, folder_name, "*.jsonl")
    files = glob.glob(search_path)
    return files[0]

logger.info("모델 로드 시작 (보호 레이어 앞뒤 1개)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True
)

# ==========================================
# 2. 데이터셋 준비
# ==========================================
logger.info("데이터셋 전처리 중...")
ds_manta = load_dataset("LGAI-EXAONE/MANTA-1M", split="train").shuffle(seed=42).select(range(NUM_SAMPLES_PER_DS))
ds_manta = ds_manta.map(lambda ex: {"text": tokenizer.apply_chat_template(ex["conversations"], add_generation_prompt=True, tokenize=False)}, remove_columns=ds_manta.column_names)

ds_gsm = load_dataset("json", data_files=get_jsonl_path("gsm8k"), split="train").shuffle(seed=42).select(range(NUM_SAMPLES_PER_DS))
ds_gsm = ds_gsm.map(lambda ex: {"text": f"Question: {ex['question']}\nAnswer: {ex['answer']}"}, remove_columns=ds_gsm.column_names)

ds_kmmlu = load_dataset("json", data_files=get_jsonl_path("kmmlu_redux"), split="train").shuffle(seed=42).select(range(NUM_SAMPLES_PER_DS))
def proc_kmmlu(ex):
    opts = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(ex['options'])])
    return {"text": f"질문: {ex['question']}\n선택지:\n{opts}\n정답: {ex['solution']}"}
ds_kmmlu = ds_kmmlu.map(proc_kmmlu, remove_columns=ds_kmmlu.column_names)

combined_ds = concatenate_datasets([ds_manta, ds_gsm, ds_kmmlu]).shuffle(seed=42)
gc.collect()

# ==========================================
# 3. v26 레시피: 보호 레이어 최소화 (앞뒤 1개)
# ==========================================
# [수정] 보호 레이어: 0번(처음)과 29번(마지막)만 제외
protected_layers = ["embed_tokens", "lm_head", "model.layers.0", "model.layers.29"]

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
        targets="Linear",
        ignore=protected_layers,
        dampening_frac=0.1
    )
]

# ==========================================
# 4. 실행 및 저장
# ==========================================
logger.info("v25 Pure GPTQ (S128, No SQ/KV, 1-Layer Protection) 시작...")
oneshot(
    model=model,
    dataset=combined_ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=len(combined_ds),
)

os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

for f in glob.glob(os.path.join(MODEL_ID, "*.py")):
    shutil.copy(f, OUT_DIR)

shutil.make_archive(os.path.join(BASE_PATH, "submit"), 'zip', WORKSPACE_DIR, "model")
logger.info("v25 수정 버전 완료!")