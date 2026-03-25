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
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# ==========================================
# [v30 맞춤형 변수 설정]
# ==========================================
DEFAULT_MODEL_PATH = "/home/jinsan/LG_Aimers_2026/03_submission/model"
DEFAULT_OUT = "/home/jinsan/LG_Aimers_2026/bench_v30_results.json"

# [중요] v30은 S=96을 사용했으므로, 96번 이후의 데이터부터 평가해야 누수가 없습니다.
CALIB_OFFSET = 64 
DEFAULT_SAMPLES = 100 # 평가할 샘플 수

def load_tokenizer(model_path):
    try:
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(f"Tokenizer load failed: {e}")

def prepare_prompt(example, dataset_name):
    if isinstance(example, dict):
        if dataset_name == 'kmmlu' and 'question' in example:
            q, opts = example.get('question', ''), example.get('options', [])
            option_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(opts)])
            return f"{q}\n\n{option_text}\n\n정답을 1, 2, 3, 4 중에서 하나만 골라주세요."
        
        if 'conversations' in example and isinstance(example['conversations'], list):
            for msg in example['conversations']:
                if msg.get('role') == 'user': return msg.get('content')
        
        for key in ('question', 'problem', 'input', 'text'):
            if key in example and example[key] is not None: return example[key]
    return str(example)

def extract_reference(example):
    if not isinstance(example, dict): return None
    for key in ('solution', 'answer', 'label', 'output', 'target'):
        if key in example and example[key] is not None: return example[key]
    if 'conversations' in example:
        for msg in reversed(example['conversations']):
            if msg.get('role') == 'assistant': return msg.get('content')
    return None

def numeric_reference(ref):
    if ref is None: return None
    s = str(ref)
    match = re.search(r"(?:정답|답)[:\s]*(\d)", s)
    if match: return float(match.group(1))
    m = re.findall(r"[-+]?\d*\.?\d+", s)
    return float(m[0]) if m else None

def evaluate_answer(gen_text, ref):
    if ref is None: return None
    num_ref = numeric_reference(ref)
    num_gen = numeric_reference(gen_text)
    if num_ref is not None and num_gen is not None:
        return float(num_gen) == float(num_ref)
    return str(ref).strip().lower() in str(gen_text).strip().lower()

def load_datasets_with_offset(samples, seed, offset):
    datasets = {}
    paths = {
        'kmmlu': '/home/jinsan/LG_Aimers_2026/datasets/kmmlu_redux/kmmlu_redux.jsonl',
        'gsm8k': '/home/jinsan/LG_Aimers_2026/datasets/gsm8k/gsm8k.jsonl'
    }

    # 1. MANTA (HF)
    try:
        ds = load_dataset("LGAI-EXAONE/MANTA-1M", split="train").shuffle(seed=seed)
        datasets['manta'] = ds.select(range(offset, offset + samples))
    except: datasets['manta'] = None

    # 2. Local JSONL (KMMLU, GSM8K)
    for name, path in paths.items():
        try:
            if os.path.exists(path):
                ds = load_dataset('json', data_files=path, split='train').shuffle(seed=seed)
                datasets[name] = ds.select(range(offset, offset + samples))
            else: datasets[name] = None
        except: datasets[name] = None

    return datasets

def _sanitize_model_dir(orig_path):
    tmpdir = tempfile.mkdtemp(prefix="model_sanitized_")
    shutil.copytree(orig_path, os.path.join(tmpdir, "model"))
    model_copy = os.path.join(tmpdir, "model")
    cfg_path = os.path.join(model_copy, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f: cfg = json.load(f)
        if "quantization_config" in cfg:
            cfg.pop("quantization_config")
            with open(cfg_path, "w") as f: json.dump(cfg, f, indent=2)
    return model_copy

def run_bench_all(model_path, samples, out_path, seed=42):
    tmp_model_dir = _sanitize_model_dir(model_path)
    tokenizer = load_tokenizer(tmp_model_dir)

    try:
        # v30의 MAX_SEQ_LEN인 1024에 맞춰 max_model_len 설정
        llm = LLM(model=model_path, gpu_memory_utilization=0.85, max_model_len=8192, trust_remote_code=True, enforce_eager=True)
    except Exception:
        model_tf = AutoModelForCausalLM.from_pretrained(tmp_model_dir, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
        class TransformersAdapter:
            def __init__(self, m, t): self.model, self.tokenizer = m, t
            def generate(self, prompts, sp):
                batch = []
                for p in prompts:
                    inputs = self.tokenizer(p, return_tensors='pt').to(self.model.device)
                    out = self.model.generate(**inputs, max_new_tokens=512)
                    text = self.tokenizer.decode(out[0], skip_special_tokens=True)
                    class O: outputs = [type('T', (object,), {'text': text, 'token_ids': out[0].tolist()})]
                    batch.append(O)
                return batch
        llm = TransformersAdapter(model_tf, tokenizer)

    sampling = SamplingParams(max_tokens=512, temperature=0.0)
    datasets = load_datasets_with_offset(samples, seed, CALIB_OFFSET)

    report = {'model_path': model_path, 'datasets': {}}
    for name, ds in datasets.items():
        if not ds: continue
        latencies, accuracies = [], []
        print(f"Evaluating {name}...")

        for idx, ex in enumerate(ds):
            prompt_content = prepare_prompt(ex, name)
            text_prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt_content}], tokenize=False, add_generation_prompt=True)

            start = time.perf_counter()
            out = llm.generate([text_prompt], sampling)
            torch.cuda.synchronize()
            
            # 토큰당 생성 속도 측정
            num_tokens = len(out[0].outputs[0].token_ids) or 1
            latency = (time.perf_counter() - start) * 1000 / num_tokens
            
            gen_text = out[0].outputs[0].text
            ref = extract_reference(ex)
            acc = evaluate_answer(gen_text, ref)
            
            latencies.append(latency)
            if acc is not None: accuracies.append(bool(acc))

        report['datasets'][name] = {
            'accuracy': sum(accuracies) / len(accuracies) if accuracies else 0,
            'latency_mean': statistics.mean(latencies),
            'samples_count': len(ds)
        }

    with open(out_path, 'w') as f: json.dump(report, f, indent=2)
    shutil.rmtree(os.path.dirname(tmp_model_dir), ignore_errors=True)
    return report

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-path', default=DEFAULT_MODEL_PATH)
    p.add_argument('--samples', type=int, default=DEFAULT_SAMPLES)
    p.add_argument('--out', default=DEFAULT_OUT)
    args = p.parse_args()
    run_bench_all(args.model_path, args.samples, args.out)
    print(f"Done! Report saved to {args.out}")

if __name__ == '__main__': main()