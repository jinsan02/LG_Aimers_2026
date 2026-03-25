# 🚀 [Portfolio] EXAONE 4.0 1.2B 경량화 및 지능 최적화 프로젝트
**Hanshin University Computer Engineering | Roh Jin-san (GPA 3.7/4.5)**

## 1. Project Overview
본 프로젝트는 LG AI Research의 **EXAONE 4.0 1.2B** 모델을 활용하여, 제한된 하드웨어 자원(L4 GPU) 환경에서 모델의 추론 속도를 극대화하고 지능(Accuracy) 손실을 최소화하는 것을 목표로 하였습니다.

* **Final Score:** **0.61537** (초기 대비 약 62% 성능 향상)
* **Key Achievement:** W4A16 양자화와 16k 컨텍스트 확장을 동시에 달성
* **Roles:** 알고리즘 선정(AWQ/GPTQ), 데이터셋 큐레이션, 메모리 최적화, 벤치마크 파이프라인 구축

---

## 2. Problem & Solution (Troubleshooting)

### 핵심 문제 1: 양자화에 따른 급격한 지능 하락
* **Problem:** 단순 GPTQ W4A16 적용 시, 수학(GSM8K) 및 논리 추론 능력이 FP16 대비 30% 이상 하락하는 현상 발생.
* **Solution:** **AWQ(Activation-aware Weight Quantization)** 알고리즘으로 전환. **Duo Scaling** 기법을 통해 활성화 값이 큰 '중요 채널'을 보호하여 지능 하락 방어 성공.
* **Result:** GSM8K와 KMMLU 점수를 복구하며 0.6점대 진입.

### 핵심 문제 2: 컨텍스트 확장과 메모리 병목
* **Problem:** 16k 확장을 위해 `max_position_embeddings`를 과도하게 설정(64k)할 경우, vLLM Serving 시 KV 캐시가 메모리를 과점유하여 추론 속도가 급격히 저하됨.
* **Solution:** 서버 사양에 맞춘 **16,384(16k) 최적 다이어트** 실시. `rope_scaling`의 factor를 2.0으로 최적화하고 `low_freq_factor` 누락에 따른 엔진 오류를 해결하여 안정성 확보.

### 핵심 문제 3: 로컬 환경의 자원 한계 (Killed 에러)
* **Problem:** 384개 이상의 샘플로 양자화 공정 시 RAM 부족으로 프로세스가 강제 종료(Killed)됨.
* **Solution:** **'Diet Calibration'** 전략 수립. `batch_size=1` 하향 및 샘플 수 최적화를 통해 로컬 자원 내에서 공정 완수.

---

## 3. Technical Deep Dive (Specialties)

### 📊 성능 비교 분석
* **Latency:** FP8 모델(10.7ms/t) 대비 W4A16 AWQ 모델(**8.4ms/t**)로 약 **21% 속도 향상**.
* **Efficiency:** `kv_cache_dtype="fp8"` 적용을 통해 긴 문맥 추론 시 메모리 대역폭 확보.



### 🏗️ 아키텍처 최적화 설정
* **Attention:** RoPE Theta 값 조절(1M → 10k)을 통한 단문 정밀도 복구.
* **Quantization Layer:** 지능의 중추인 `embed_tokens`와 `lm_head`를 양자화 대상에서 제외(Ignore List)하여 유창성 유지.

---

## 4. Retrospective & Growth
이 프로젝트를 통해 모델의 파라미터 수만큼이나 **데이터 밸런스(Manta/GSM/KMMLU)**와 **하드웨어 친화적 설정**이 실무에서 얼마나 중요한지 깨달았습니다. 특히 4비트 환경에서도 16k 문맥을 안정적으로 처리해낸 경험은 향후 대규모 언어 모델 서빙 역량에 큰 자산이 될 것입니다.
