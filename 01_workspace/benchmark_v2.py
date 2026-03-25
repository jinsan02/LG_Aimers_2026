import argparse
import time
import os
import json
import re
import statistics
import torch
import shutil
import tempfile
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets

# ==========================================
# 1. v47 기본 설정 (MBPP 제거 상태 유지)
# ==========================================
DEFAULT_MODEL_PATH = "/home/jinsan/LG_Aimers_2026/03_submission/model"
DEFAULT_OUT = "/home/jinsan/LG_Aimers_2026/bench_v47_report.json"
DEFAULT_SAMPLES = 100 
DEFAULT_OFFSET = 200 # 리더보드 점수와 겹치지 않게 설정

def load_tokenizer(model_path):
    try:
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(f"Tokenizer load failed: {e}")

def prepare_prompt(example, dataset_name):
    if isinstance(example, dict):
        if dataset_name == 'kmmlu' and 'question' in example and 'options' in example:
            q = example.get('question', '')
            opts = example.get('options', [])
            option_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(opts)])
            return f"{q}\n\n{option_text}\n\n정답을 1, 2, 3, 4 중에서 하나만 골라주세요."
        if 'conversations' in example and isinstance(example['conversations'], list):
            for msg in example['conversations']:
                if msg.get('role') == 'user': return msg.get('content')
        for key in ('question', 'problem', 'input', 'text'):
            if key in example and example[key] is not None: return example[key]
    return str(example)

def extract_reference(example, dataset_name):
    if not isinstance(example, dict): return None
    if 'conversations' in example and isinstance(example['conversations'], list):
        for msg in reversed(example['conversations']):
            if msg.get('role') == 'assistant': return msg.get('content')
    for key in ('solution', 'answer', 'label', 'output', 'target', 'responses', 'ground_truth'):
        if key in example and example[key] is not None: return example[key]
    return None

# ==========================================
# [수정] GSM8K 및 KMMLU 통합 숫자 추출 로직 (v3 참고)
# ==========================================
def numeric_reference(ref):
    if ref is None or isinstance(ref, list): return None
    s = str(ref).strip()
    
    # 1. GSM8K 전용 패턴 (#### 뒤의 숫자 추출)
    gsm_match = re.search(r"####\s*([-+]?\d*\.?\d+)", s)
    if gsm_match: return float(gsm_match.group(1))
    
    # 2. 범용 정답 라벨 추출 (정답: X, 답: X 등)
    # v3의 (\d)를 ([-+]?\d*\.?\d+)로 확장하여 소수점 및 큰 수 대응
    label_match = re.search(r"(?:정답|답|answer)[:\s]*([-+]?\d*\.?\d+)", s, re.IGNORECASE)
    if label_match: return float(label_match.group(1))
    
    # 3. 텍스트 내 첫 번째 숫자 추출 (v3 로직)
    m = re.findall(r"[-+]?\d*\.?\d+", s)
    if m:
        try:
            return float(m[0])
        except: return None
    return None

def evaluate_answer(gen_text, ref, dataset_name):
    if ref is None: return None
    
    # 숫자 비교 시도
    num_ref = numeric_reference(ref)
    num_gen = numeric_reference(gen_text)
    
    if num_ref is not None and num_gen is not None:
        return float(num_gen) == float(num_ref)
    
    # 숫자 비교 실패 시 텍스트 매칭 (v2/v3 공통)
    def norm(text): return re.sub(r"\s+", " ", str(text)).strip().lower()
    return norm(ref) in norm(gen_text)

# ==========================================
# 4. 데이터 로딩 및 실행 로직 (v2 유지)
# ==========================================
def load_datasets(samples, seed, offset=0):
    datasets = {}
    # 1. MANTA-1M
    try:
        ds_manta = load_dataset("LGAI-EXAONE/MANTA-1M", split="train")
        datasets['manta'] = ds_manta.shuffle(seed=seed).select(range(offset, offset + min(len(ds_manta) - offset, samples)))
    except: datasets['manta'] = None
    # 2. KMMLU-Redux
    kmmlu_path = '/home/jinsan/LG_Aimers_2026/datasets/kmmlu_redux/kmmlu_redux.jsonl'
    try:
        ds_km = load_dataset('json', data_files=kmmlu_path, split='train')
        datasets['kmmlu'] = ds_km.shuffle(seed=seed).select(range(offset, offset + min(len(ds_km) - offset, samples)))
    except: datasets['kmmlu'] = None
    # 3. GSM8K
    gsm8k_path = '/home/jinsan/LG_Aimers_2026/datasets/gsm8k/gsm8k.jsonl'
    try:
        ds_gsm = load_dataset('json', data_files=gsm8k_path, split='train')
        datasets['gsm8k'] = ds_gsm.shuffle(seed=seed).select(range(offset, offset + min(len(ds_gsm) - offset, samples)))
    except: datasets['gsm8k'] = None
    
    return datasets

def run_bench_all(model_path, gpu_mem_util, max_model_len, samples, out_path, offset=0, seed=42, quantization='compressed-tensors'):
    tokenizer = load_tokenizer(model_path)
    try:
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=max_model_len,
            trust_remote_code=True,
            enforce_eager=True,
            quantization=quantization,
            kv_cache_dtype="auto"
        )
    except Exception as e:
        print(f"[ERROR] vLLM 로드 실패: {e}")
        return

    sampling = SamplingParams(max_tokens=512, temperature=0.0)
    datasets = load_datasets(samples, seed, offset)
    report = {'model_path': model_path, 'offset_used': offset, 'datasets': {}}

    for name, ds in datasets.items():
        if ds is None: continue
        latencies, accuracies, recs = [], [], []
        print(f"🔍 Evaluating {name}...")

        for idx, ex in enumerate(ds):
            prompt_content = prepare_prompt(ex, name)
            text_prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt_content}], tokenize=False, add_generation_prompt=True)

            start = time.perf_counter()
            out = llm.generate([text_prompt], sampling)
            torch.cuda.synchronize()
            end = time.perf_counter()

            gen_text = out[0].outputs[0].text
            tokens = len(out[0].outputs[0].token_ids) or 1
            latency = ((end - start) / tokens) * 1000
            latencies.append(latency)

            ref = extract_reference(ex, name)
            acc = evaluate_answer(gen_text, ref, name)
            if acc is not None: accuracies.append(bool(acc))
            recs.append({'idx': idx + offset, 'correct': acc, 'latency': latency})

        report['datasets'][name] = {
            'count': len(recs),
            'latency_ms_per_token': {'mean': statistics.mean(latencies)},
            'accuracy': {'accuracy': sum(accuracies) / len(accuracies) if accuracies else None},
        }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"✅ 벤치마크 완료! 결과 저장: {out_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-path', default=DEFAULT_MODEL_PATH)
    p.add_argument('--samples', type=int, default=DEFAULT_SAMPLES)
    p.add_argument('--offset', type=int, default=DEFAULT_OFFSET)
    p.add_argument('--out', default=DEFAULT_OUT)
    args = p.parse_args()

    run_bench_all(args.model_path, 0.85, 8192, args.samples, args.out, offset=args.offset)

if __name__ == '__main__':
    main()