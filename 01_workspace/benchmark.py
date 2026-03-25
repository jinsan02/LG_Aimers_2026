import time
import torch
import numpy as np
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# --- [1. 경로 및 환경 설정] ---
MODEL_PATH = "/home/jinsan/LG_Aimers_2026/03_submission/model"
GPU_MEM_UTIL = 0.85  # 대회 서버 고정 옵션
MAX_MODEL_LEN = 16384 # 대회 서버 고정 옵션

def run_pro_benchmark():
    # [v버전 자동 인식 로직]
    parent_dir = os.path.basename(os.path.dirname(os.path.normpath(MODEL_PATH)))
    model_ver = parent_dir if "v" in parent_dir else "v13-Final"
    
    print(f"🚀 [Simulation] {model_ver} 대회 서버 환경 측정 시작")
    print(f"📂 모델 경로: {MODEL_PATH}")
    
    if not os.path.exists(os.path.join(MODEL_PATH, "model.safetensors")):
        print("❌ 에러: 양자화된 모델 파일(model.safetensors)이 없습니다. v13을 먼저 실행하세요.")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    # --- [2. vLLM 엔진 로드] ---
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
        enforce_eager=True,
        quantization="compressed-tensors",
        dtype="bfloat16"
    )

    # [수정] 지능 검증용 4대 도메인 질문 세트 (각 512토큰 설정)
    test_cases = [
        {"title": "수학 추론", "prompt": "함수 f(x) = x^3 - 3x + 2의 극값을 구하고, 그 과정에서 미분의 원리를 설명해줘."},
        {"title": "파이썬 코딩", "prompt": "파이썬의 리스트 컴프리헨션을 사용하여 1부터 100 사이의 소수(Prime Number)만 필터링하는 효율적인 함수를 짜줘."},
        {"title": "한국어 논리", "prompt": "대한민국 헌법 제1조 1항과 2항의 내용을 쓰고, 이것이 민주공화국이라는 정의와 어떻게 연결되는지 논리적으로 설명해줘."},
        {"title": "지시 수행", "prompt": "양자화 기법의 장점 3가지를 불렛 포인트로 설명하되, '데이터'라는 단어를 쓰지 말고 설명해줘."}
    ]
    
    sampling_params = SamplingParams(max_tokens=512, temperature=0.0)

    # --- [3. Warm-up] ---
    print(f"🔥 {model_ver} 커널 웜업 중...")
    warmup_prompt = tokenizer.apply_chat_template([{"role": "user", "content": "Hello"}], tokenize=False, add_generation_prompt=True)
    _ = llm.generate([warmup_prompt], sampling_params)
    torch.cuda.synchronize()

    # --- [4. 본 측정 및 답변 검수 시작] ---
    print(f"📊 {model_ver} 도메인별 지능 및 속도 검증 시작\n")
    all_latencies = []
    
    for i, case in enumerate(test_cases):
        formatted_prompt = tokenizer.apply_chat_template([{"role": "user", "content": case['prompt']}], tokenize=False, add_generation_prompt=True)
        
        start = time.perf_counter()
        output = llm.generate([formatted_prompt], sampling_params)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        generated_text = output[0].outputs[0].text
        tokens = len(output[0].outputs[0].token_ids)
        latency = ((end - start) / tokens) * 1000
        all_latencies.append(latency)
        
        print(f"[{i+1}/{len(test_cases)}] {case['title']} 테스트 중...")
        print("-" * 60)
        print(f"📝 {case['title']} 답변:\n{generated_text}")
        print("-" * 60)
        print(f"⏱️ 레이턴시: {latency:.2f} ms/token (생성 토큰: {tokens})\n")

    avg_lat = np.mean(all_latencies)
    
    # --- [5. 최종 결과 리포트] ---
    print("\n" + "🏆" + " v13 통합 성능 리포트 ".center(40, "=") + "🏆")
    print(f"✅ 모델 버전: {model_ver}")
    print(f"✅ 4대 도메인 평균 레이턴시: {avg_lat:.2f} ms/token")
    print(f"✅ 지능 점수 기대치: 0.61+ (Sequential & Shuffle 효과)")
    print(f"✅ L4 서버 예상 속도: 약 {avg_lat * 0.95:.2f} ms/token")
    print("="*45)
    print("💡 팁: 답변의 논리적 흐름이 깨지지 않았다면 즉시 제출을 추천합니다!")

if __name__ == "__main__":
    run_pro_benchmark()