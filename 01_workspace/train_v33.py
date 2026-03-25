import os, torch, shutil, glob
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from compressed_tensors.quantization import (
    QuantizationScheme, QuantizationArgs, QuantizationType, QuantizationStrategy
)

# ==========================================
# [v33 설정] Full W4A16 + KMMLU 강화 (1:1:2)
# ==========================================
BASE_S = 100        # 기본 단위 샘플 수
MAX_SEQ_LEN = 768   # L = 768

BASE_PATH = "/home/jinsan/LG_Aimers_2026"
MODEL_ID = os.path.join(BASE_PATH, "00_base_model")
SUBMISSION_DIR = os.path.join(BASE_PATH, "03_submission")
OUT_DIR = os.path.join(SUBMISSION_DIR, "model")

# 필수 보호 요소
ignore_list = ["embed_tokens", "lm_head"] 

# 1. 캘리브레이션 데이터 로드 (Manta 1 : GSM8K 1 : KMMLU 2)
def load_calib_data():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f"📦 데이터 로드 중... (Manta:{BASE_S}, GSM8K:{BASE_S}, KMMLU:{BASE_S*2})")
    
    # MANTA-1M (1배수)
    ds_manta = load_dataset("LGAI-EXAONE/MANTA-1M", split="train").shuffle(seed=42).select(range(BASE_S))
    ds_manta = ds_manta.map(lambda x: {"text": tokenizer.apply_chat_template(x["conversations"], tokenize=False, add_generation_prompt=True)}, remove_columns=ds_manta.column_names)
    
    # GSM8K (1배수)
    ds_gsm = load_dataset("json", data_files=os.path.join(BASE_PATH, "datasets/gsm8k/gsm8k.jsonl"), split="train").shuffle(seed=42).select(range(BASE_S))
    ds_gsm = ds_gsm.map(lambda x: {"text": f"Question: {x['question']}\nAnswer: {x['answer']}"}, remove_columns=ds_gsm.column_names)
    
    # KMMLU-Redux (2배수)
    ds_kmmlu = load_dataset("json", data_files=os.path.join(BASE_PATH, "datasets/kmmlu_redux/kmmlu_redux.jsonl"), split="train").shuffle(seed=42).select(range(BASE_S * 2))
    ds_kmmlu = ds_kmmlu.map(lambda x: {"text": f"질문: {x['question']}\n정답: {x['solution']}"}, remove_columns=ds_kmmlu.column_names)
    
    return concatenate_datasets([ds_manta, ds_gsm, ds_kmmlu]).shuffle(seed=42)

# 2. 양자화 실행
def run_quant():
    print("🚀 Full W4A16 양자화 공정 시작 (Damping: 0.1, Max Embedding: 16384)")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    
    # [설정 반영] Max Position Embeddings 수정
    model.config.max_position_embeddings = 16384
    # original_max_position_embeddings는 8192 유지 (베이스라인 기준점)

    # W4A16 설정
    scheme_w4 = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4, 
            type=QuantizationType.INT, 
            strategy=QuantizationStrategy.CHANNEL, 
            dynamic=False
        ),
    )

    # GPTQ 레시피 (Damping 0.1 적용)
    recipe = [
        GPTQModifier(
            config_groups={"group_w4": scheme_w4}, 
            targets="Linear", 
            ignore=ignore_list,
            dampening_frac=0.1  # Damping 0.1 설정
        )
    ]

    oneshot(
        model=model, 
        dataset=load_calib_data(), 
        recipe=recipe, 
        max_seq_length=MAX_SEQ_LEN, 
        num_calibration_samples=BASE_S * 4
    )

    # 3. 저장 및 압축
    if os.path.exists(SUBMISSION_DIR): shutil.rmtree(SUBMISSION_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)
    
    model.save_pretrained(OUT_DIR, save_compressed=True, format="marlin")
    
    # 토크너 및 필수 스크립트 복사
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.save_pretrained(OUT_DIR)
    for f in glob.glob(os.path.join(MODEL_ID, "*.py")):
        shutil.copy(f, OUT_DIR)
    
    # 제출용 zip 생성 (model 폴더 포함 구조)
    shutil.make_archive(os.path.join(SUBMISSION_DIR, "submit"), 'zip', SUBMISSION_DIR, 'model')
    print(f"✅ v33 공정 완료! 파일 위치: {SUBMISSION_DIR}/submit.zip")

if __name__ == "__main__":
    run_quant()