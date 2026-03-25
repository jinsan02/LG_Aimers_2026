import os, torch, shutil, glob
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from compressed_tensors.quantization import (
    QuantizationScheme, QuantizationArgs, QuantizationType, QuantizationStrategy
)

# ==========================================
# 1. 환경 및 하이퍼파라미터 설정 (S=96으로 조정)
# ==========================================
BASE_S = 96        # 메모리 안정성을 위해 96으로 설정
MAX_SEQ_LEN = 768   

BASE_PATH = "/home/jinsan/LG_Aimers_2026"
MODEL_ID = os.path.join(BASE_PATH, "00_base_model")
SUBMISSION_DIR = os.path.join(BASE_PATH, "03_submission")
OUT_DIR = os.path.join(SUBMISSION_DIR, "model")

# 지능 방어용 보호 레이어
ignore_list = ["embed_tokens", "lm_head", "model.layers.0", "model.layers.1", "model.layers.28", "model.layers.29"]

# ==========================================
# 2. 캘리브레이션 데이터 로드 (1:1:2 비율)
# ==========================================
def load_calib_data():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f"📦 데이터 로드 중... (Total Samples: {BASE_S * 4})")
    
    ds_manta = load_dataset("LGAI-EXAONE/MANTA-1M", split="train").shuffle(seed=42).select(range(BASE_S))
    ds_manta = ds_manta.map(lambda x: {"text": tokenizer.apply_chat_template(x["conversations"], tokenize=False, add_generation_prompt=True)}, remove_columns=ds_manta.column_names)
    
    ds_gsm = load_dataset("json", data_files=os.path.join(BASE_PATH, "datasets/gsm8k/gsm8k.jsonl"), split="train").shuffle(seed=42).select(range(BASE_S))
    ds_gsm = ds_gsm.map(lambda x: {"text": f"Question: {x['question']}\nAnswer: {x['answer']}"}, remove_columns=ds_gsm.column_names)
    
    ds_kmmlu = load_dataset("json", data_files=os.path.join(BASE_PATH, "datasets/kmmlu_redux/kmmlu_redux.jsonl"), split="train").shuffle(seed=42).select(range(BASE_S * 2))
    ds_kmmlu = ds_kmmlu.map(lambda x: {"text": f"질문: {x['question']}\n정답: {x['solution']}"}, remove_columns=ds_kmmlu.column_names)
    
    return concatenate_datasets([ds_manta, ds_gsm, ds_kmmlu]).shuffle(seed=42)

# ==========================================
# 3. 양자화 실행 (FP8 + ActOrder 적용, KV-Cache 제거)
# ==========================================
def run_quantization():
    print("🚀 v35: FP8 + ActOrder 양자화 시작 (KV-Cache 양자화 제외)")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    
    model.config.max_position_embeddings = 16384

    # [FP8 스키마 설정]
    scheme_fp8 = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=8, type=QuantizationType.FLOAT, strategy=QuantizationStrategy.TENSOR, dynamic=False
        ),
        input_activations=QuantizationArgs(
            num_bits=8, type=QuantizationType.FLOAT, strategy=QuantizationStrategy.TENSOR, dynamic=True
        )
    )

    # [레시피 구성] actorder 옵션 추가 및 kv_cache_scheme 제거
    recipe = [
        GPTQModifier(
            config_groups={"group_fp8": scheme_fp8}, 
            targets="Linear", 
            ignore=ignore_list,
            actorder="per_channel" # 활성화 값 크기에 따른 양자화 순서 최적화 활성화
        )
    ]

    oneshot(
        model=model, 
        dataset=load_calib_data(), 
        recipe=recipe, 
        max_seq_length=MAX_SEQ_LEN, 
        num_calibration_samples=BASE_S * 4
    )

    # ==========================================
    # 4. 저장 및 압축
    # ==========================================
    if os.path.exists(SUBMISSION_DIR): shutil.rmtree(SUBMISSION_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)
    
    model.save_pretrained(OUT_DIR, save_compressed=True)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.save_pretrained(OUT_DIR)
    for f in glob.glob(os.path.join(MODEL_ID, "*.py")):
        shutil.copy(f, OUT_DIR)
    
    shutil.make_archive(os.path.join(SUBMISSION_DIR, "submit"), 'zip', SUBMISSION_DIR, 'model')
    print(f"✅ v35 공정 완료! 제출 파일: {SUBMISSION_DIR}/submit.zip")

if __name__ == "__main__":
    run_quantization()