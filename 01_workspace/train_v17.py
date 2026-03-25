import os, torch, shutil
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs, QuantizationType, QuantizationStrategy

# 1. 환경 및 경로 설정
MODEL_ID = "/home/jinsan/LG_Aimers_2026/00_base_model"
WORKSPACE_DIR = "/home/jinsan/LG_Aimers_2026/03_submission"
OUT_DIR = os.path.join(WORKSPACE_DIR, "model")

# [V17 전략] 최적의 샘플 수(512)와 지식 보호를 위한 셔플링 재도입
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 512

print(f"[INFO] v17(The Safety High-Score) 실행 중 (S512 + Shuffle + G128 + B128 + D0.01)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto")

# [데이터] 셔플링 재도입: v16(No-Shuffle)보다 v15(Shuffle)에서 지식 왜곡이 덜했던 결과 반영
print(f"[INFO] 데이터셋 셔플링 및 로드 중...")
ds = load_dataset("LGAI-EXAONE/MANTA-1M", split="train")
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["conversations"], add_generation_prompt=True, tokenize=False)}

ds = ds.map(preprocess)

# 2. v17 레시피: "8ms대 초고속 레이턴시 및 고정밀 보정"
custom_scheme = QuantizationScheme(
    targets=["Linear"],
    weights=QuantizationArgs(
        num_bits=4,          # W4 (Weight 4-bit)
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.GROUP,
        group_size=128,      # [속도 사수] 6 vCPU 병목 해소를 위해 128로 설정
        symmetric=True,
        dynamic=False,
        actorder="static"
    )
)

recipe = [
    GPTQModifier(
        config_groups={"group_0": custom_scheme},
        ignore=["embed_tokens", "lm_head"],
        block_size=128,      # [핵심] 128로 높여 10ms 실격 리스크 원천 차단
        dampening_frac=0.01, # [지능 사수] v11.6 성공 수치로 수학/코딩 정밀도 확보
        # sequential_targets 제거 (One-shot): 1.2B 모델의 지식 표류 현상 방지
    )
]

print(f"[INFO] v17 양자화 시작 (One-shot / High-Speed 모드)...")
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# 3. 저장 및 제출 규격 압축
if os.path.exists(OUT_DIR): shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

shutil.make_archive(os.path.join(WORKSPACE_DIR, "..", "submit_v17_Final"), "zip", WORKSPACE_DIR, "model")
print(f"✅ v17 생성 완료! 8.5ms대의 속도와 0.6 이상의 점수를 기대합니다.")