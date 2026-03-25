import os, torch, shutil, glob, logging
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from compressed_tensors.quantization import (
    QuantizationScheme, QuantizationArgs, QuantizationType, QuantizationStrategy
)

# Colab 노트북 환경 변수 적용
BASE_MODEL_DIR = "/content/drive/MyDrive/comp/base_model"
SUBMISSION_DIR = "/content/drive/MyDrive/comp/submissions"

# 임시 빌드 경로 (zip 압축 전 구조 잡기용)
TEMP_BUILD_DIR = os.path.join(SUBMISSION_DIR, "temp_v37_build")
OUT_DIR        = os.path.join(TEMP_BUILD_DIR, "model")

# 데이터셋 경로 (노트북 상위 셀에서 다운로드/준비된 경로)
GSM8K_FILE = "/content/gsm8k_cache/gsm8k_train.jsonl"
KMMLU_FILE = "/content/KMMLU-Redux/data/kmmlu_redux.jsonl"

CONFIG = {
    "samples_common": 128,  # Manta, GSM8K 각 128개
    "samples_kmmlu": 256,   # KMMLU 256개 (총 S=512 물량 공세)
    "max_seq_len": 1024,    # GSM8K 0.82점을 위한 필수 깊이
    "dampening_frac": 0.1,  # 수치 안정성
    "seed": 42
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EXAONE_V37_ABSOLUTE")

# [1. 데이터 로드 로직] - S=512 풀 파워
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True)

def load_calib_data():
    s_c, s_k = CONFIG["samples_common"], CONFIG["samples_kmmlu"]

    # MANTA (대화)
    ds_manta = load_dataset("LGAI-EXAONE/MANTA-1M", split="train").shuffle(seed=CONFIG["seed"]).select(range(s_c))
    ds_manta = ds_manta.map(lambda x: {"text": tokenizer.apply_chat_template(x["conversations"], tokenize=False, add_generation_prompt=True)}, remove_columns=ds_manta.column_names)

    # GSM8K (수학)
    # 로컬 파일이 있으면 로드, 없으면 HuggingFace 로드
    if os.path.exists(GSM8K_FILE):
        ds_gsm = load_dataset("json", data_files=GSM8K_FILE, split="train").shuffle(seed=CONFIG["seed"]).select(range(s_c))
    else:
        logger.warning("Local GSM8K file not found, loading from HuggingFace main...")
        ds_gsm = load_dataset("gsm8k", "main", split="train").shuffle(seed=CONFIG["seed"]).select(range(s_c))

    ds_gsm = ds_gsm.map(lambda x: {"text": f"Question: {x['question']}\nAnswer: {x['answer']}"}, remove_columns=ds_gsm.column_names)

    # KMMLU (지식 - 물량 강화)
    ds_kmmlu = load_dataset("json", data_files=KMMLU_FILE, split="train").shuffle(seed=CONFIG["seed"]).select(range(s_k))
    ds_kmmlu = ds_kmmlu.map(lambda x: {"text": f"질문: {x['question']}\n정답: {x['solution']}"}, remove_columns=ds_kmmlu.column_names)

    return concatenate_datasets([ds_manta, ds_gsm, ds_kmmlu]).shuffle(seed=CONFIG["seed"])

# [2. FP8 지능 사수 공정]
def run_quantization():
    logger.info("최종 승부수: FP8 지능 방어 절대주의 공정 개시 (S=512, L=1024)")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_DIR, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)

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

    # [3. 저장 및 압축]
    if os.path.exists(TEMP_BUILD_DIR): shutil.rmtree(TEMP_BUILD_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    logger.info(f"Saving model to {OUT_DIR}...")
    model.save_pretrained(OUT_DIR, save_compressed=True)
    tokenizer.save_pretrained(OUT_DIR)

    # 필요한 .py 파일 등 복사 (없을 수도 있음)
    for f in glob.glob(os.path.join(BASE_MODEL_DIR, "*.py")):
        shutil.copy(f, OUT_DIR)

    # Zip 생성
    submit_zip_path = os.path.join(SUBMISSION_DIR, "submit_v37_absolute")
    shutil.make_archive(submit_zip_path, 'zip', TEMP_BUILD_DIR, 'model')
    logger.info(f"✅ v37 Victory Absolute 공정 완료! Saved to {submit_zip_path}.zip")

if __name__ == "__main__":
    run_quantization()