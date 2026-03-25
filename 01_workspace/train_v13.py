import os, torch, shutil
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs, QuantizationType, QuantizationStrategy

# 1. 환경 및 데이터 설정 (진산 님의 로컬 경로 유지)
MODEL_ID = "/home/jinsan/LG_Aimers_2026/00_base_model"
WORKSPACE_DIR = "/home/jinsan/LG_Aimers_2026/03_submission"
OUT_DIR = os.path.join(WORKSPACE_DIR, "model")

NUM_CALIBRATION_SAMPLES = 1024  # [강화] 샘플 수 상향
MAX_SEQUENCE_LENGTH = 512

print(f"[INFO] v13 최종본 로드 중 (Shuffle & S1024 & G64)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto")

# [핵심] 셔플링을 통한 데이터 대표성 확보
ds = load_dataset("LGAI-EXAONE/MANTA-1M", split="train")
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))
ds = ds.map(lambda x: {"text": tokenizer.apply_chat_template(x["conversations"], add_generation_prompt=True, tokenize=False)})

# 2. v13 정밀 레시피
custom_scheme = QuantizationScheme(
    targets=["Linear"],
    weights=QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.GROUP,
        group_size=64,       # 해상도 64 고정
        symmetric=True,
        dynamic=False,
        actorder="static"
    )
)

recipe = [
    GPTQModifier(
        config_groups={"group_0": custom_scheme},
        ignore=["embed_tokens", "lm_head"], # 임베딩 보호 (Float16 유지 확인됨)
        block_size=64,
        dampening_frac=0.1,
        # [최종 수정] 직접 확인한 클래스 이름 반영
        sequential_targets=["Exaone4DecoderLayer"] 
    )
]

# 3. 양자화 실행 및 저장
print(f"[INFO] v13 초정밀 양자화 시작 (Sequential 모드 활성)...")
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

if os.path.exists(OUT_DIR): shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

shutil.make_archive(os.path.join(WORKSPACE_DIR, "..", "submit_v13_Final"), "zip", WORKSPACE_DIR, "model")
print(f"✅ v13 최종본 생성 완료! 셔플된 1024개 샘플과 레이어별 보정이 적용되었습니다.")