import os, torch, shutil, glob
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from compressed_tensors.quantization import (
    QuantizationScheme, QuantizationArgs, QuantizationType, QuantizationStrategy
)

# ==========================================
# 1. 환경 및 하이퍼파라미터 설정
# ==========================================
BASE_PATH = "/home/jinsan/LG_Aimers_2026"
MODEL_ID = os.path.join(BASE_PATH, "00_base_model")
SUBMISSION_DIR = os.path.join(BASE_PATH, "03_submission")
OUT_DIR = os.path.join(SUBMISSION_DIR, "model")

DS_S = 128          # 데이터셋당 샘플 수 (총 384개)
MAX_SEQ_LEN = 768   # 데이터셋별 평균 길이를 고려한 최적 길이

# [샌드위치 구역 정의]
# 임베딩과 헤드만 BF16 유지
ignore_list = ["embed_tokens", "lm_head"] 

# FP8 보호막: 0~7번, 22~29번 (지능 방어용 16개 레이어)
fp8_layers = [f"model.layers.{i}" for i in range(0, 8)] + \
             [f"model.layers.{i}" for i in range(22, 30)]

# W4A16 압축: 8~21번 (속도 폭격용 14개 레이어)
w4_layers = [f"model.layers.{i}" for i in range(8, 22)]

# ==========================================
# 2. 캘리브레이션 데이터 로드 (Manta + GSM8K + KMMLU)
# ==========================================
def load_calib_data():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f"📦 데이터 로드 중... (비율 1:1:1, 총 샘플: {DS_S * 3})")
    
    # MANTA-1M
    ds_manta = load_dataset("LGAI-EXAONE/MANTA-1M", split="train").shuffle(seed=42).select(range(DS_S))
    ds_manta = ds_manta.map(lambda x: {"text": tokenizer.apply_chat_template(x["conversations"], tokenize=False, add_generation_prompt=True)}, remove_columns=ds_manta.column_names)
    
    # GSM8K (수학)
    ds_gsm = load_dataset("json", data_files=os.path.join(BASE_PATH, "datasets/gsm8k/gsm8k.jsonl"), split="train").shuffle(seed=42).select(range(DS_S))
    ds_gsm = ds_gsm.map(lambda x: {"text": f"Question: {x['question']}\nAnswer: {x['answer']}"}, remove_columns=ds_gsm.column_names)
    
    # KMMLU-Redux (한국어 지식)
    ds_kmmlu = load_dataset("json", data_files=os.path.join(BASE_PATH, "datasets/kmmlu_redux/kmmlu_redux.jsonl"), split="train").shuffle(seed=42).select(range(DS_S))
    ds_kmmlu = ds_kmmlu.map(lambda x: {"text": f"질문: {x['question']}\n정답: {x['solution']}"}, remove_columns=ds_kmmlu.column_names)
    
    return concatenate_datasets([ds_manta, ds_gsm, ds_kmmlu]).shuffle(seed=42)

# ==========================================
# 3. 양자화 실행 (Two-Pass GPTQ)
# ==========================================
def run_quantization():
    print("🚀 모델 로드 및 샌드위치 양자화 시작...")
    
    # 모델 로드 (config 수정 포함)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    # 문맥 길이를 8k로 최적화하여 메모리 확보
    model.config.max_position_embeddings = 16384
    
    # [스키마 A] FP8 설정 (지능용)
    scheme_fp8 = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(num_bits=8, type=QuantizationType.FLOAT, strategy=QuantizationStrategy.TENSOR, dynamic=False),
        input_activations=QuantizationArgs(num_bits=8, type=QuantizationType.FLOAT, strategy=QuantizationStrategy.TENSOR, dynamic=True),
    )

    # [스키마 B] W4A16 설정 (속도용)
    scheme_w4 = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(num_bits=4, type=QuantizationType.INT, strategy=QuantizationStrategy.CHANNEL, dynamic=False),
    )

    # [레시피] 각 구역이 서로를 침범하지 않게 교차 ignore 설정
    recipe = [
        GPTQModifier(config_groups={"group_fp8": scheme_fp8}, targets="Linear", ignore=ignore_list + w4_layers),
        GPTQModifier(config_groups={"group_w4": scheme_w4}, targets="Linear", ignore=ignore_list + fp8_layers)
    ]

    # 양자화 실행
    oneshot(
        model=model, 
        dataset=load_calib_data(), 
        recipe=recipe, 
        max_seq_length=MAX_SEQ_LEN, 
        num_calibration_samples=DS_S * 3
    )

    # ==========================================
    # 4. 저장 및 제출 파일 생성
    # ==========================================
    if os.path.exists(SUBMISSION_DIR): shutil.rmtree(SUBMISSION_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)
    
    model.save_pretrained(OUT_DIR, save_compressed=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.save_pretrained(OUT_DIR)
    
    # 필요한 전용 파이썬 파일들 복사
    for f in glob.glob(os.path.join(MODEL_ID, "*.py")):
        shutil.copy(f, OUT_DIR)
    
    # 03_submission 폴더를 submit.zip으로 압축
    shutil.make_archive(os.path.join(SUBMISSION_DIR, "submit"), 'zip', SUBMISSION_DIR, 'model')
    print(f"✅ 모든 공정 완료! 제출 파일 위치: {SUBMISSION_DIR}/submit.zip")

if __name__ == "__main__":
    run_quantization()