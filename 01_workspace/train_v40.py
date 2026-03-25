import os, torch, shutil, glob, gc
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from compressed_tensors.quantization import (
    QuantizationScheme, QuantizationArgs, QuantizationType, QuantizationStrategy
)

# ==========================================
# 1. 환경 및 하이퍼파라미터 설정 (S=128, L=768)
# ==========================================
BASE_S = 128         # 128 * 3 = 총 384 샘플 (v35와 동일 토큰 버짓 유지)
MAX_SEQ_LEN = 768   

BASE_PATH = "/home/jinsan/LG_Aimers_2026"
MODEL_ID = os.path.join(BASE_PATH, "00_base_model")
SUBMISSION_DIR = os.path.join(BASE_PATH, "03_submission")
OUT_DIR = os.path.join(SUBMISSION_DIR, "model")

# [요청 사항] 임베딩과 헤드 레이어만 보호
ignore_list = ["embed_tokens", "lm_head"]

# ==========================================
# 2. 3개 데이터셋 균등 로드 (Manta, GSM, KMMLU)
# ==========================================
def load_calib_data():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f"📦 v49 데이터 로드: Manta, GSM, KMMLU 각 {BASE_S}개 (총 384개)")
    
    # 1. Manta (대화)
    ds_manta = load_dataset("LGAI-EXAONE/MANTA-1M", split="train").shuffle(seed=42).select(range(BASE_S))
    ds_manta = ds_manta.map(lambda x: {"text": tokenizer.apply_chat_template(x["conversations"], tokenize=False, add_generation_prompt=True)}, remove_columns=ds_manta.column_names)
    
    # 2. GSM8K (수학)
    ds_gsm = load_dataset("json", data_files=os.path.join(BASE_PATH, "datasets/gsm8k/gsm8k.jsonl"), split="train").shuffle(seed=42).select(range(BASE_S))
    ds_gsm = ds_gsm.map(lambda x: {"text": f"Question: {x['question']}\nAnswer: {x['answer']}"}, remove_columns=ds_gsm.column_names)
    
    # 3. KMMLU-Redux (한국어 지식)
    ds_kmmlu = load_dataset("json", data_files=os.path.join(BASE_PATH, "datasets/kmmlu_redux/kmmlu_redux.jsonl"), split="train").shuffle(seed=42).select(range(BASE_S))
    ds_kmmlu = ds_kmmlu.map(lambda x: {"text": f"질문: {x['question']}\n정답: {x['solution']}"}, remove_columns=ds_kmmlu.column_names)
    
    return concatenate_datasets([ds_manta, ds_gsm, ds_kmmlu]).shuffle(seed=42)

# ==========================================
# 3. AWQ 정밀 공정 및 16k 확장
# ==========================================
def run_quantization():
    print(f"🚀 v49: AWQ 정밀 모드 시작 (n_grid=40, ratio=0.2)")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )

    # [컨텍스트 16384 확장]
    model.config.max_position_embeddings = 16384
    
    # [요청 사항] n_grid=40, ratio=0.2 추가 적용
    # ※ 주의: 라이브러리 버전에 따라ValidationError 발생 시 해당 두 라인 제거 필요
    recipe = [
        AWQModifier(
            targets="Linear",
            ignore=ignore_list,
            scheme="W4A16_ASYM",
            duo_scaling=True,
        )
    ]

    calib_data = load_calib_data()

    oneshot(
        model=model, 
        dataset=calib_data, 
        recipe=recipe, 
        max_seq_length=MAX_SEQ_LEN, 
        num_calibration_samples=len(calib_data)
    )

    # 메모리 정리 (Killed 방지)
    del calib_data
    gc.collect()
    torch.cuda.empty_cache()

    # 4. 저장 및 압축
    if os.path.exists(SUBMISSION_DIR): shutil.rmtree(SUBMISSION_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR, save_compressed=True)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.save_pretrained(OUT_DIR)
    for f in glob.glob(os.path.join(MODEL_ID, "*.py")): shutil.copy(f, OUT_DIR)
    
    shutil.make_archive(os.path.join(SUBMISSION_DIR, "submit"), 'zip', SUBMISSION_DIR, 'model')
    print(f"✅ v49 공정 완료! 3개 데이터셋 기반 16k AWQ 모델이 생성되었습니다.")

if __name__ == "__main__":
    run_quantization()