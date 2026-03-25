import os, torch, shutil, glob, gc, json
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

# ==========================================
# 1. 환경 및 안정성 우선 하이퍼파라미터 (다이어트)
# ==========================================
BASE_S = 96          # [수정] 샘플당 64개 (총 192개)로 RAM 부하 감소
MAX_SEQ_LEN = 768    # [수정] 768 -> 512로 줄여 캘리브레이션 안정성 확보
BASE_PATH = "/home/jinsan/LG_Aimers_2026"
MODEL_ID = os.path.join(BASE_PATH, "00_base_model")
OUT_DIR = os.path.join(BASE_PATH, "03_submission/model")

ignore_list = ["embed_tokens", "lm_head"]

# ==========================================
# 2. 데이터셋 로딩 (샘플링 비중 유지)
# ==========================================
def load_calib_data():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f"📦 v58 다이어트 로드: 각 {BASE_S}개 (총 {BASE_S * 3}개)")
    
    ds_manta = load_dataset("LGAI-EXAONE/MANTA-1M", split="train").shuffle(seed=42).select(range(BASE_S))
    ds_manta = ds_manta.map(lambda x: {"text": tokenizer.apply_chat_template(x["conversations"], tokenize=False, add_generation_prompt=True)}, remove_columns=ds_manta.column_names)
    
    ds_gsm = load_dataset("json", data_files=os.path.join(BASE_PATH, "datasets/gsm8k/gsm8k.jsonl"), split="train").shuffle(seed=42).select(range(BASE_S))
    ds_gsm = ds_gsm.map(lambda x: {"text": f"Question: {x['question']}\nAnswer: {x['answer']}"}, remove_columns=ds_gsm.column_names)
    
    ds_kmmlu = load_dataset("json", data_files=os.path.join(BASE_PATH, "datasets/kmmlu_redux/kmmlu_redux.jsonl"), split="train").shuffle(seed=42).select(range(BASE_S))
    ds_kmmlu = ds_kmmlu.map(lambda x: {"text": f"질문: {x['question']}\n정답: {x['solution']}"}, remove_columns=ds_kmmlu.column_names)
    
    return concatenate_datasets([ds_manta, ds_gsm, ds_kmmlu]).shuffle(seed=42)

# ==========================================
# 3. 양자화 및 컨텍스트 최적화 (v57 핵심 설정 계승)
# ==========================================
def run_quantization():
    print(f"🚀 v58: Killed 에러 방지 모드 시작 (Batch=1, Seq=512)")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.bfloat16, trust_remote_code=True
    )

    # [중요] 16k 최적화 및 RoPE 오류 수정 설정 계승
    model.config.max_position_embeddings = 16384 
    model.config.rope_scaling = {
        "rope_type": "llama3",
        "factor": 2.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192
    }
    model.config.rope_theta = 10000.0
    if not hasattr(model.config, "quantization_config"): model.config.quantization_config = {}
    model.config.quantization_config["kv_cache_dtype"] = "fp8"

    recipe = [AWQModifier(targets="Linear", ignore=ignore_list, scheme="W4A16_ASYM", duo_scaling=True)]

    calib_data = load_calib_data()

    # [Killed 방지] Batch Size를 1로 낮춰 메모리 피크를 억제합니다.
    oneshot(
        model=model, 
        dataset=calib_data, 
        recipe=recipe, 
        batch_size=1,                  # 4 -> 1로 하향
        max_seq_length=MAX_SEQ_LEN, 
        num_calibration_samples=len(calib_data)
    )

    # 4. 저장 및 정리
    if os.path.exists(OUT_DIR): shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR, save_compressed=True)
    AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True).save_pretrained(OUT_DIR)
    for f in glob.glob(os.path.join(MODEL_ID, "*.py")): shutil.copy(f, OUT_DIR)
    
    print(f"✅ v58 공정 성공! 메모리 제한 내에서 최적화 모델을 완성했습니다.")

if __name__ == "__main__":
    run_quantization()