# 📑 Project History: EXAONE 4.0 1.2B Optimization
** Hanshin University | Computer Engineering | Roh Jin-san**

## 🏆 Final Result
* **Public Score:** **0.61537**
* **Model:** EXAONE 4.0 1.2B (v58 Final optimized)
* **Key Achievement:** W4A16 양자화와 16k 컨텍스트 확장을 동시에 달성하면서 지능 점수 0.6 돌파

---

## 🚀 Optimization Roadmap

### Phase 1: 기반 구축 (v30 ~ v40)
* **주요 작업:** 초기 8k 컨텍스트 유지 및 GPTQ W4A16 양자화 시도.
* **평가 데이터:** Manta, GSM8K, KMMLU, MBPP 4종 세트 구성.
* **문제점:** GPTQ 적용 시 GSM8K 추론 능력이 급격히 하락하는 현상 발생.

### Phase 2: 지능 복구 및 알고리즘 전환 (v41 ~ v50)
* **알고리즘 변경:** GPTQ에서 **AWQ (Activation-aware Weight Quantization)**로 전환.
* **Duo Scaling 적용:** 활성화 값의 분포를 고려하여 중요한 가중치를 보호함으로써 수학(GSM8K) 및 논리 능력을 FP8 수준으로 복구.
* **데이터셋 정제:** MBPP를 제외하고 3개 핵심 데이터셋(Manta, GSM, KMMLU)에 128개씩 균등 배분하여 한국어 지식과 수학에 집중.

### Phase 3: 컨텍스트 확장 및 정밀 튜닝 (v51 ~ v57)
* **문맥 확장:** 8192 -> 16384 (16k) 확장 완료.
* **RoPE Scaling 최적화:**
    * `factor`: 16.0 → 2.0 (16k 타겟 최적화)
    * `rope_theta`: 1,000,000 → 10,000 (단문 정밀도 복구)
    * `low_freq_factor`: 1.0 (vLLM KeyError 해결)
* **추론 가속:** `kv_cache_dtype="fp8"` 및 `enforce_eager=False` 적용으로 토큰당 생성 시간 단축.

### Phase 4: 자원 한계 극복 (v58 Final)
* **RAM 부족 대응:** 캘리브레이션 중 `Killed` 에러 발생에 따른 **'Diet 공정'** 실시.
* **안정화 설정:** 샘플 수 조정 및 `batch_size=1` 설정을 통해 로컬 환경에서 안정적인 양자화 완수.

---

## 🛠 Technical Stack
| Category | Specification |
| :--- | :--- |
| **Base Model** | EXAONE 4.0 1.2B Instruct |
| **Quantization** | AWQ W4A16 (Asymmetric, Group 128) |
| **Context Length** | 16,384 (16k) |
| **Optimization** | Duo Scaling, FP8 KV Cache, CUDA Graph |
| **Library** | vLLM, llmcompressor, transformers |

---

## 💡 Lessons Learned
1.  **소형 모델의 민감도:** 1.2B 모델은 파라미터가 적어 양자화 오차에 매우 민감함. 단순 Weight 절삭보다 AWQ의 **Duo Scaling**이 지능 방어에 핵심적임을 확인.
2.  **컨텍스트의 명과 암:** `max_position_embeddings`를 무조건 크게(64k 등) 잡는 것보다 서버 사양(16k)에 맞춰 최적화하는 것이 메모리 안정성과 속도 면에서 훨씬 유리함.
3.  **데이터 밸런스:** 특정 데이터셋(MBPP)이 지능 점수를 깎아먹는다면 과감히 제거하고 핵심 데이터(GSM, KMMLU)에 집중하는 전략이 유효함.

