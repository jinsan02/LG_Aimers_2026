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

# [v23 핵심 규격: 지능과 속도의 최적 합의점]
# L=768 (v21 벤치에서 검증된 지능 복구 길이)
# S=384 (도메인당 128개, RAM 32GB 환경의 안전 마지노선)
NUM_SAMPLES_PER_DS = 128 
MAX_SEQUENCE_LENGTH = 768

print(f"[INFO] v24 KV-Ultimate 실행 중 | L={MAX_SEQUENCE_LENGTH}, S={NUM_SAMPLES_PER_DS * 3}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16, # BF16 로드로 FP8과의 수치적 정렬 최적화
    device_map="auto"
)

# ==========================================
# 2. 다도메인 데이터셋 로드 (정확도 방어의 핵심)
# ==========================================
print(f"[INFO] 다도메인 데이터셋 캘리브레이션 준비...")

# (1) MANTA (일반 지시어)
ds_manta = load_dataset("LGAI-EXAONE/MANTA-1M", split="train").shuffle(seed=42).select(range(NUM_SAMPLES_PER_DS))
ds_manta = ds_manta.map(lambda ex: {"text": tokenizer.apply_chat_template(ex["conversations"], add_generation_prompt=True, tokenize=False)}, remove_columns=ds_manta.column_names)

# (2) GSM8K (수학적 추론)
gsm8k_file = get_jsonl_path("gsm8k")
ds_gsm = load_dataset("json", data_files=gsm8k_file, split="train").shuffle(seed=42).select(range(NUM_SAMPLES_PER_DS))
ds_gsm = ds_gsm.map(lambda ex: {"text": f"Question: {ex['question']}\nAnswer: {ex['answer']}"}, remove_columns=ds_gsm.column_names)

# (3) KMMLU-Redux (한국어 지식 및 상식)
kmmlu_file = get_jsonl_path("kmmlu_redux")
ds_kmmlu = load_dataset("json", data_files=kmmlu_file, split="train").shuffle(seed=42).select(range(NUM_SAMPLES_PER_DS))
def proc_kmmlu(ex):
    opts = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(ex['options'])])
    return {"text": f"질문: {ex['question']}\n선택지:\n{opts}\n정답: {ex['solution']}"}
ds_kmmlu = ds_kmmlu.map(proc_kmmlu, remove_columns=ds_kmmlu.column_names)

# 데이터 병합 (균형 잡힌 지능 주입)
combined_ds = concatenate_datasets([ds_manta, ds_gsm, ds_kmmlu]).shuffle(seed=42)

# ==========================================
# 3. v23 레시피: FP8 W8A8 + KV Cache (방법 1)
# ==========================================
# 가중치 및 활성화를 위한 FP8 스킴 (W8A8)
fp8_scheme = QuantizationScheme(
    targets=["Linear"],
    weights=QuantizationArgs(num_bits=8, type=QuantizationType.FLOAT, strategy=QuantizationStrategy.TENSOR, symmetric=True, dynamic=False),
    input_activations=QuantizationArgs(num_bits=8, type=QuantizationType.FLOAT, strategy=QuantizationStrategy.TENSOR, symmetric=True, dynamic=True)
)

# [수정] GPTQModifier 내부에서 kv_cache_scheme을 처리하도록 하여 Pydantic 에러 방지
recipe = [
    GPTQModifier(
        config_groups={"group_0": fp8_scheme},
        # KV Cache를 8비트로 압축하여 추론 시 메모리 대역폭 병목을 제거합니다.
        kv_cache_scheme=QuantizationArgs(num_bits=8, type=QuantizationType.FLOAT, strategy=QuantizationStrategy.TENSOR, symmetric=True),
        ignore=["embed_tokens", "lm_head"],
        targets="Linear",
        dampening_frac=0.1 # 소형 모델의 수치 안정성 확보
    )
]

# ==========================================
# 4. 실행 및 저장
# ==========================================
print(f"[INFO] FP8 Weights + Act + KV-Cache 캘리브레이션 시작 (예상 15분 내외)...")
oneshot(
    model=model,
    dataset=combined_ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=len(combined_ds),
)

print("[INFO] 양자화 공정 완료. 결과물 저장 및 압축 중...")
os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

# [검증 완료] 제출용 ZIP 파일 생성 (base_name 형식 준수)
zip_name = os.path.join(BASE_PATH, "submit")
shutil.make_archive(base_name=zip_name, format='zip', root_dir=WORKSPACE_DIR, base_dir="model")

print(f"[INFO] v23 KV-Ultimate 성공적으로 완료되었습니다.")
print(f"[INFO] 최종 제출 파일: {zip_name}.zip")