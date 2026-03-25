import os, torch, shutil, glob, gc
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from compressed_tensors.quantization import (
    QuantizationScheme, QuantizationArgs, QuantizationType, QuantizationStrategy
)

# ==========================================
# 1. 환경 및 하이퍼파라미터 설정 (S=48로 추가 하향)
# ==========================================
BASE_S = 96        # 48 * 4 = 총 192 샘플 (RAM 안전 확보용)
MAX_SEQ_LEN = 768  

BASE_PATH = "/home/jinsan/LG_Aimers_2026"
MODEL_ID = os.path.join(BASE_PATH, "00_base_model")
MBPP_PATH = os.path.join(BASE_PATH, "datasets/mbpp_local")
SUBMISSION_DIR = os.path.join(BASE_PATH, "03_submission")
OUT_DIR = os.path.join(SUBMISSION_DIR, "model")

ignore_list = ["embed_tokens", "lm_head"]

# ==========================================
# 2. 데이터셋 로드 함수
# ==========================================
def load_calib_data():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f"📦 v41 데이터 로드: Manta:{BASE_S}, GSM:{BASE_S}, KMMLU:{BASE_S}, MBPP:{BASE_S}")
    
    ds_manta = load_dataset("LGAI-EXAONE/MANTA-1M", split="train")
    manta_count = min(BASE_S, len(ds_manta))
    ds_manta = ds_manta.shuffle(seed=42).select(range(manta_count))
    ds_manta = ds_manta.map(lambda x: {"text": tokenizer.apply_chat_template(x["conversations"], tokenize=False, add_generation_prompt=True)}, remove_columns=ds_manta.column_names)
    
    ds_gsm = load_dataset("json", data_files=os.path.join(BASE_PATH, "datasets/gsm8k/gsm8k.jsonl"), split="train")
    gsm_count = min(BASE_S, len(ds_gsm))
    ds_gsm = ds_gsm.shuffle(seed=42).select(range(gsm_count))
    ds_gsm = ds_gsm.map(lambda x: {"text": f"Question: {x['question']}\nAnswer: {x['answer']}"}, remove_columns=ds_gsm.column_names)
    
    ds_kmmlu = load_dataset("json", data_files=os.path.join(BASE_PATH, "datasets/kmmlu_redux/kmmlu_redux.jsonl"), split="train")
    km_count = min(BASE_S, len(ds_kmmlu))
    ds_kmmlu = ds_kmmlu.shuffle(seed=42).select(range(km_count))
    ds_kmmlu = ds_kmmlu.map(lambda x: {"text": f"질문: {x['question']}\n정답: {x['solution']}"}, remove_columns=ds_kmmlu.column_names)
    
    ds_mbpp = load_dataset("google-research-datasets/mbpp", "sanitized", split="train", cache_dir=MBPP_PATH)
    mbpp_count = min(BASE_S, len(ds_mbpp))
    if mbpp_count <= 0:
        raise RuntimeError(f"MBPP(train) 데이터 로드 실패 또는 샘플 없음 (path={MBPP_PATH})")
    ds_mbpp = ds_mbpp.shuffle(seed=42).select(range(mbpp_count))
    ds_mbpp = ds_mbpp.map(lambda x: {"text": f"Task: {x['prompt']}\nCode: {x['code']}"}, remove_columns=ds_mbpp.column_names)
    
    return concatenate_datasets([ds_manta, ds_gsm, ds_kmmlu, ds_mbpp]).shuffle(seed=42)

# ==========================================
# 3. 양자화 실행 및 메모리 관리
# ==========================================
def run_quantization():
    print("🚀 v41: RAM 최적화 W4A16 공정 시작 (S=48)")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    
    scheme_w4 = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(num_bits=4, type=QuantizationType.INT, strategy=QuantizationStrategy.GROUP, group_size=128, symmetric=True)
    )

    recipe = [GPTQModifier(config_groups={"group_w4": scheme_w4}, targets="Linear", ignore=ignore_list, actorder="static", dampening_frac=0.1)]

    # 캘리브레이션 데이터 로드
    calib_data = load_calib_data()

    # 양자화 수행
    oneshot(
        model=model, 
        dataset=calib_data, 
        recipe=recipe, 
        max_seq_length=MAX_SEQ_LEN, 
        num_calibration_samples=len(calib_data)
    )

    # [메모리 확보 핵심 로직]
    # 데이터셋 등 불필요한 객체 삭제 후 RAM 청소
    del calib_data
    gc.collect()
    torch.cuda.empty_cache()
    print("🧹 메모리 정리 완료. 모델 저장을 시작합니다...")

    # 저장 및 압축
    if os.path.exists(SUBMISSION_DIR): shutil.rmtree(SUBMISSION_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # save_compressed=True 시 발생하는 RAM 스파이크 방지
    model.save_pretrained(OUT_DIR, save_compressed=True)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.save_pretrained(OUT_DIR)
    for f in glob.glob(os.path.join(MODEL_ID, "*.py")): shutil.copy(f, OUT_DIR)
    
    shutil.make_archive(os.path.join(SUBMISSION_DIR, "submit"), 'zip', SUBMISSION_DIR, 'model')
    print(f"✅ v41 공정 최종 완료!")

if __name__ == "__main__":
    run_quantization()