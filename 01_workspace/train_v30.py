import os
import torch
import shutil
import glob
import logging
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# ==========================================
# [MASTER CONFIG] - W4A16 설정 전용
# ==========================================
CONFIG = {
    "model_id": "/home/jinsan/LG_Aimers_2026/00_base_model",
    "save_dir": "/home/jinsan/LG_Aimers_2026/03_submission",
    "samples_per_ds": 64,      # S = 128
    "max_seq_len": 1024,        # L = 768
    "scheme": "W4A16",         # 가중치 4비트 양자화
    "dampening_frac": 0.1,     # 수치 안정성 향상
    "seed": 42,
    "ignore_layers": ["embed_tokens", "lm_head"] # 필수 보호 레이어
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EXAONE_v33_W4A16")

# [1. 데이터 로드 로직] - Manta, GSM8K, KMMLU 믹스
def get_calib_dataset(tokenizer):
    s = CONFIG["samples_per_ds"]
    logger.info(f"데이터셋 준비 중 (S={s}, L={CONFIG['max_seq_len']})...")
    
    ds_manta = load_dataset("LGAI-EXAONE/MANTA-1M", split="train").shuffle(seed=CONFIG["seed"]).select(range(s))
    ds_manta = ds_manta.map(lambda x: {"text": tokenizer.apply_chat_template(x["conversations"], tokenize=False, add_generation_prompt=True)}, remove_columns=ds_manta.column_names)
    
    gsm_path = "/home/jinsan/LG_Aimers_2026/datasets/gsm8k/gsm8k.jsonl"
    ds_gsm = load_dataset("json", data_files=gsm_path, split="train").shuffle(seed=CONFIG["seed"]).select(range(s))
    ds_gsm = ds_gsm.map(lambda x: {"text": f"Question: {x['question']}\nAnswer: {x['answer']}"}, remove_columns=ds_gsm.column_names)
    
    kmmlu_path = "/home/jinsan/LG_Aimers_2026/datasets/kmmlu_redux/kmmlu_redux.jsonl"
    ds_kmmlu = load_dataset("json", data_files=kmmlu_path, split="train").shuffle(seed=CONFIG["seed"]).select(range(s))
    ds_kmmlu = ds_kmmlu.map(lambda x: {"text": f"질문: {x['question']}\n정답: {x['solution']}"}, remove_columns=ds_kmmlu.column_names)
    
    return concatenate_datasets([ds_manta, ds_gsm, ds_kmmlu]).shuffle(seed=CONFIG["seed"])

# [2. 메인 양자화 공정]
def run_quantization():
    out_dir = os.path.join(CONFIG["save_dir"], "model")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_id"], trust_remote_code=True)
    
    # 모델 로드 (W4A16은 BF16 상태에서 진행)
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_id"], 
        device_map="auto", 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    )

    # GPTQ W4A16 레시피 적용
    recipe = [GPTQModifier(
        scheme=CONFIG["scheme"], 
        targets="Linear", 
        ignore=CONFIG["ignore_layers"],
        dampening_frac=CONFIG["dampening_frac"]
    )]

    # 양자화 실행
    oneshot(
        model=model, 
        dataset=get_calib_dataset(tokenizer), 
        recipe=recipe, 
        max_seq_length=CONFIG["max_seq_len"], 
        num_calibration_samples=CONFIG["samples_per_ds"] * 3
    )

    # [3. 저장 및 파일 정리] - 가이드 준수
    if os.path.exists(CONFIG["save_dir"]): shutil.rmtree(CONFIG["save_dir"])
    os.makedirs(out_dir, exist_ok=True)
    
    model.save_pretrained(out_dir, save_compressed=True)
    tokenizer.save_pretrained(out_dir)
    
    # 필수 설계도 파일(.py) 복사
    for f in glob.glob(os.path.join(CONFIG["model_id"], "*.py")):
        shutil.copy(f, out_dir)
    
    # [4. 최종 압축] - submit.zip 생성
    shutil.make_archive(
        base_name=os.path.join("/home/jinsan/LG_Aimers_2026", "submit"), 
        format='zip', 
        root_dir=CONFIG["save_dir"], 
        base_dir='model'
    )
    logger.info("✅ v31 W4A16 공정 및 압축 완료!")

if __name__ == "__main__":
    run_quantization()