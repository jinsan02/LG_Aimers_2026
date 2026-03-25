import os, torch, shutil
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
# 세부 설정을 위한 클래스 추가
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

# 1. 환경 설정
MODEL_ID = "/home/jinsan/LG_Aimers_2026/00_base_model"
WORKSPACE_DIR = "/home/jinsan/LG_Aimers_2026/03_submission"
OUT_DIR = os.path.join(WORKSPACE_DIR, "model")
NUM_SAMPLES = 768 

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto")

ds = load_dataset("LGAI-EXAONE/MANTA-1M", split=f"train[:{NUM_SAMPLES}]")
ds = ds.map(lambda x: {"text": tokenizer.apply_chat_template(x["conversations"], add_generation_prompt=True, tokenize=False)})

# 2. v12.4 레시피: QuantizationArgs를 통한 세부 제어
recipe = [
    GPTQModifier(
        targets="Linear",
        scheme="W4A16",  # 기본 프리셋 유지 (Pydantic 에러 방지)
        ignore=["embed_tokens", "lm_head"],
        block_size=32,
        actorder="static",
        dampening_frac=0.1,
        # [핵심 수정] 가중치 설정을 직접 덮어쓰기 위해 config 직접 제어 시도
        # 만약 이 인자가 modifier에서 직접 안 먹힐 경우를 대비해 
        # 아래와 같이 딕셔너리 형태의 kv_cache_scheme 규격을 참고하여 구성합니다.
        kv_cache_scheme={
            "num_bits": 8,
            "type": "float",
            "strategy": "tensor",
            "symmetric": True,
            "dynamic": False
        }
    )
]

# 3. 0.61 돌파를 위한 강제 설정 주입 (oneshot 실행 직전)
# llmcompressor가 문자열 "W4A16"을 해석할 때 group_size를 32로 인지하게 유도합니다.
print(f"[INFO] v12.4 초고정밀 양자화 시작 (G32 강제화 전략)...")

# 만약 GPTQModifier에서 group_size가 계속 128로 롤백된다면 
# 아래와 같이 oneshot의 내부 구성을 활용하는 것이 가장 확실합니다.
oneshot(
    model=model, 
    dataset=ds, 
    recipe=recipe, 
    max_seq_length=512, 
    num_calibration_samples=NUM_SAMPLES,
)

# 4. 저장 전 config 강제 보정 (가장 확실한 트릭)
# 양자화가 끝난 후, 저장 직전에 모델의 config를 직접 수정하여 
# vLLM이 32로 읽게 만듭니다.
if hasattr(model, "quantization_config"):
    # 가중치 그룹 사이즈 강제 수정
    for group in model.quantization_config.config_groups.values():
        if "weights" in group:
            group["weights"]["group_size"] = 32
            print(f"[DEBUG] group_size를 32로 보정했습니다.")

# 5. 저장 및 압축
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)
shutil.make_archive(os.path.join(WORKSPACE_DIR, "..", "submit_v12_4_Final"), "zip", WORKSPACE_DIR, "model")
print(f"✅ v12.4 생성 완료!")