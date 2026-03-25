import os
import torch
import shutil
import glob
import logging
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from compressed_tensors.quantization import (
    QuantizationScheme, QuantizationArgs, QuantizationType, QuantizationStrategy
)

# [1. 마스터 설정] - 고점 모델 분석 결과 반영
BASE_PATH = "/home/jinsan/LG_Aimers_2026"
MODEL_ID = os.path.join(BASE_PATH, "00_base_model")
SUBMISSION_DIR = os.path.join(BASE_PATH, "03_submission")
OUT_DIR = os.path.join(SUBMISSION_DIR, "model")

CONFIG = {
    "samples_common": 96,  # Manta, GSM8K 각 128개
    "samples_kmmlu": 192,   # 약점인 한국어 지식(KMMLU) 2배 강화 (총 S=512)
    "max_seq_len": 768,    # GSM8K 0.82점을 위한 필수 길이
    "dampening_frac": 0.1,  # v24에서 검증된 수치 안정성 옵션
    "seed": 42
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EXAONE_V35_VICTORY")

# [2. 고품질 멀티 도메인 데이터셋 준비]
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

def load_calib_data():
    logger.info(f"데이터 로드 시작: S={CONFIG['samples_common']*2 + CONFIG['samples_kmmlu']}, L={CONFIG['max_seq_len']}")
    
    # MANTA (일상 대화)
    ds_manta = load_dataset("LGAI-EXAONE/MANTA-1M", split="train").shuffle(seed=CONFIG["seed"]).select(range(CONFIG["samples_common"]))
    ds_manta = ds_manta.map(lambda x: {"text": tokenizer.apply_chat_template(x["conversations"], tokenize=False, add_generation_prompt=True)}, remove_columns=ds_manta.column_names)
    
    # GSM8K (수학적 추론)
    gsm_path = os.path.join(BASE_PATH, "datasets/gsm8k/gsm8k.jsonl")
    ds_gsm = load_dataset("json", data_files=gsm_path, split="train").shuffle(seed=CONFIG["seed"]).select(range(CONFIG["samples_common"]))
    ds_gsm = ds_gsm.map(lambda x: {"text": f"Question: {x['question']}\nAnswer: {x['answer']}"}, remove_columns=ds_gsm.column_names)
    
    # KMMLU (한국어 지식 - v5의 성공 요인 반영)
    kmmlu_path = os.path.join(BASE_PATH, "datasets/kmmlu_redux/kmmlu_redux.jsonl")
    ds_kmmlu = load_dataset("json", data_files=kmmlu_path, split="train").shuffle(seed=CONFIG["seed"]).select(range(CONFIG["samples_kmmlu"]))
    ds_kmmlu = ds_kmmlu.map(lambda x: {"text": f"질문: {x['question']}\n정답: {x['solution']}"}, remove_columns=ds_kmmlu.column_names)
    
    return concatenate_datasets([ds_manta, ds_gsm, ds_kmmlu]).shuffle(seed=CONFIG["seed"])

# [3. FP8 양자화 공정] - v24의 지능 보존 방식 적용
logger.info("모델 로드 및 FP8 양자화 준비...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)

scheme = QuantizationScheme(
    targets=["Linear"],
    weights=QuantizationArgs(num_bits=8, type=QuantizationType.FLOAT, strategy=QuantizationStrategy.TENSOR, dynamic=False),
    input_activations=QuantizationArgs(num_bits=8, type=QuantizationType.FLOAT, strategy=QuantizationStrategy.TENSOR, dynamic=True),
)

recipe = [GPTQModifier(
    config_groups={"group_0": scheme}, 
    targets="Linear", 
    ignore=["embed_tokens", "lm_head"],
    dampening_frac=CONFIG["dampening_frac"]
)]

oneshot(
    model=model, 
    dataset=load_calib_data(), 
    recipe=recipe, 
    max_seq_length=CONFIG["max_seq_len"], 
    num_calibration_samples=CONFIG["samples_common"]*2 + CONFIG["samples_kmmlu"]
)

# [4. 파일 정리 및 데이콘 규격 저장]
if os.path.exists(SUBMISSION_DIR):
    shutil.rmtree(SUBMISSION_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

logger.info("최종 모델 및 필수 파일 저장 중...")
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

# EXAONE 추론을 위한 필수 커스텀 코드(.py) 복사
for f in glob.glob(os.path.join(MODEL_ID, "*.py")):
    shutil.copy(f, OUT_DIR)

# [5. 최종 제출용 압축 (submit.zip)]
shutil.make_archive(
    base_name=os.path.join(BASE_PATH, "submit"), 
    format="zip",
    root_dir=SUBMISSION_DIR,
    base_dir="model"
)

logger.info(f"✅ v31 Victory 공정 완료! 제출 파일: {BASE_PATH}/submit.zip")