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

# --- [1. 경로 및 다이어트 하이퍼파라미터] ---
BASE_PATH = "/home/jinsan/LG_Aimers_2026"
MODEL_ID = f"{BASE_PATH}/00_base_model"
OUT_DIR = f"{BASE_PATH}/03_submission/model"
DATASET_ID = "LGAI-EXAONE/MANTA-1M"

DTYPE = torch.float16 # FP16 정밀도 유지

# [V8 전략: 극강의 다이어트]
NUM_FT_SAMPLES = 2000      # 10,000 -> 2,000 (데이터 양 대폭 축소)
NUM_CALIB_SAMPLES = 1024   # 양자화 정밀도는 유지
MAX_SEQ_LEN = 512

LEARNING_RATE = 2e-5       # 1e-4 -> 2e-5 (학습 강도 하향)
MAX_STEPS = 50             # 150 -> 50 (망각 방지)

if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# --- [2. 모델 및 긴 답변 위주 데이터 로드] ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=DTYPE, device_map="auto", trust_remote_code=True
)

# 답변 길이를 기준으로 정렬하여 상위 2,000개만 사용 (단답형 방지)
full_ds = load_dataset(DATASET_ID, split="train")
def get_length(example): return len(example["conversations"][-1]["content"])
sorted_ds = full_ds.map(lambda x: {"len": get_length(x)}).sort("len", reverse=True)
diet_ds = sorted_ds.select(range(NUM_FT_SAMPLES))

def tokenize_fn(examples):
    full_text = [tokenizer.apply_chat_template(conv, tokenize=False) for conv in examples["conversations"]]
    return tokenizer(full_text, truncation=True, max_length=MAX_SEQ_LEN, padding="max_length")
tokenized_ds = diet_ds.map(tokenize_fn, batched=True, remove_columns=diet_ds.column_names)

# --- [3. LoRA 회복 학습 함수] ---
def run_recovery_training(model, step_name):
    print(f"🚀 [v8-Diet] {step_name} 지능 복구 시작 (Step: {MAX_STEPS})...")
    model.enable_input_require_grads() 

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_ds,
        args=TrainingArguments(
            per_device_train_batch_size=1, 
            gradient_accumulation_steps=8,
            max_steps=MAX_STEPS, 
            learning_rate=LEARNING_RATE, 
            fp16=True, 
            logging_steps=5, 
            optim="paged_adamw_32bit", 
            output_dir="./tmp_v8"
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    trainer.train()
    merged_model = model.merge_and_unload()
    torch.cuda.empty_cache()
    return merged_model

# --- [4. v8 사이클 실행] ---
# Step 1: 10% Pruning -> Recovery
oneshot(model=model, dataset=tokenized_ds.select(range(NUM_CALIB_SAMPLES)), 
        recipe=[WandaPruningModifier(sparsity=0.10, targets=["Linear"])])
model = run_recovery_training(model, "10% Recovery")

# Step 2: 20% Pruning -> Recovery
oneshot(model=model, dataset=tokenized_ds.select(range(NUM_CALIB_SAMPLES)), 
        recipe=[WandaPruningModifier(sparsity=0.20, targets=["Linear"])])
model = run_recovery_training(model, "20% Recovery")

# Step 3: 최종 GPTQ 양자화
oneshot(model=model, dataset=tokenized_ds.select(range(NUM_CALIB_SAMPLES)),
        recipe=[GPTQModifier(targets=["Linear"], scheme="W4A16", ignore=["embed_tokens", "lm_head"])])

# --- [5. 저장 및 submit.zip 고정 생성] ---
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

# 아카이브용 이름 대신 무조건 submit.zip으로 생성
shutil.make_archive(
    base_name=f"{BASE_PATH}/submit", 
    format="zip",
    root_dir=f"{BASE_PATH}/03_submission",
    base_dir="model"
)
print(f"✅ v8 학습 완료 및 submit.zip 생성 완료!")