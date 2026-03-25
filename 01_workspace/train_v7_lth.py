import os
import torch
import shutil
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.pruning.wanda import WandaPruningModifier

# --- [1. 경로 및 정밀도 설정] ---
MODEL_ID = "/home/jinsan/LG_Aimers_2026/00_base_model"
# 최종 제출용 model 폴더로 경로 고정
OUT_DIR = "/home/jinsan/LG_Aimers_2026/03_submission/model"
DATASET_ID = "LGAI-EXAONE/MANTA-1M"

# 리더보드 결과 반영: FP16(0.5044)이 우세하므로 float16 설정
DTYPE = torch.float16 

NUM_FT_SAMPLES = 10000    # 지능 복구용 샘플 수 상향
NUM_CALIB_SAMPLES = 1024  # 양자화 정밀도용 샘플 수 강화
MAX_SEQ_LEN = 512

# 기존 model 폴더 초기화 (제출 구조 엄수)
if os.path.exists(OUT_DIR):
    print(f"[INFO] 기존 {OUT_DIR} 삭제 중...")
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# --- [2. 모델 및 데이터 로드] ---
print(f"[INFO] 모델 로드 중 ({DTYPE})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=DTYPE, device_map="auto", trust_remote_code=True
)

# MANTA-1M 데이터셋 준비
ds = load_dataset(DATASET_ID, split=f"train[:{NUM_FT_SAMPLES}]")
def tokenize_fn(examples):
    full_text = [tokenizer.apply_chat_template(conv, tokenize=False) for conv in examples["conversations"]]
    return tokenizer(full_text, truncation=True, max_length=MAX_SEQ_LEN, padding="max_length")
tokenized_ds = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)

# --- [3. LoRA 회복 학습 함수 (RuntimeError 수정 완료)] ---
def run_recovery_training(model, step_name):
    print(f"🚀 [v7-Pro] {step_name} 지능 복구 시작 (FP16 모드)...")
    
    # [에러 해결 핵심] Gradient Checkpointing 사용 시 필수 설정
    model.enable_input_require_grads() 

    # MLP 레이어까지 포함하여 지능 복구 극대화
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable() # VRAM 8GB 절약용

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_ds,
        args=TrainingArguments(
            per_device_train_batch_size=1, 
            gradient_accumulation_steps=8, # 실제 배치 사이즈 8 효과
            max_steps=150,                 # 지능 복구에 최적화된 스텝
            learning_rate=1e-4, 
            fp16=True, 
            logging_steps=10, 
            optim="paged_adamw_32bit",     # 메모리 효율적 옵티마이저
            output_dir="./tmp_v7"
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    trainer.train()
    
    # 학습된 LoRA 가중치를 베이스 모델에 병합하여 프루닝/양자화 준비
    merged_model = model.merge_and_unload()
    torch.cuda.empty_cache()
    return merged_model

# --- [4. v7 필승 사이클 실행] ---

# Step 1: 10% Pruning -> 1차 회복 학습
print("✂️ Step 1: 10% Wanda Pruning...")
oneshot(model=model, dataset=tokenized_ds.select(range(NUM_CALIB_SAMPLES)), 
        recipe=[WandaPruningModifier(sparsity=0.10, targets=["Linear"])])
model = run_recovery_training(model, "10% Recovery")

# Step 2: 20% Pruning -> 2차 회복 학습 (로또 티켓 가설 적용)
print("✂️ Step 2: 20% Wanda Pruning...")
oneshot(model=model, dataset=tokenized_ds.select(range(NUM_CALIB_SAMPLES)), 
        recipe=[WandaPruningModifier(sparsity=0.20, targets=["Linear"])])
model = run_recovery_training(model, "20% Recovery")

# Step 3: 최종 GPTQ 양자화 (W4A16)
print("💎 Step 3: 최종 정밀 GPTQ Quantization...")
oneshot(
    model=model, 
    dataset=tokenized_ds.select(range(NUM_CALIB_SAMPLES)),
    recipe=[GPTQModifier(targets=["Linear"], scheme="W4A16", ignore=["embed_tokens", "lm_head"])]
)

# --- [5. 저장 및 제출 압축] ---
print(f"[INFO] v7 최종 모델 저장 중: {OUT_DIR}")
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

# submit_v7.zip 자동 생성
shutil.make_archive(
    base_name="/home/jinsan/LG_Aimers_2026/submit_v7", 
    format="zip",
    root_dir="/home/jinsan/LG_Aimers_2026/03_submission",
    base_dir="model"
)
print(f"✅ v7 최종 모델 생성 및 압축 완료! ~/LG_Aimers_2026/submit_v7.zip")