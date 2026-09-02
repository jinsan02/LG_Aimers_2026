# Project History: EXAONE 4.0 1.2B Optimization

Hanshin University | Computer Engineering | Roh Jin-san

## 버전 체계

- **v37**: 외부 제출 파일명에 남은 최고 평가 제출 버전
- **v58**: 이후 내부 개발이 계속된 뒤의 최종 버전

두 번호는 같은 시점의 단순 대소 비교가 아닙니다. 최고 평가 기록과 최종 내부 개발 산출물을 분리해 관리합니다.

## 검증된 최고 평가 제출

- 제출 파일: submit_v37_absolute.zip
- Public Score: **0.6153749319**
- 평가 시간: **9분 45초**
- 생성 코드: 01_workspace/final.py
- 양자화: FP8 W8A8
- calibration: MANTA 128 + GSM8K 128 + KMMLU 256
- calibration max sequence length: 1024
- 제출 기록 context: 16,384

final.py는 외부 base_model 상태를 사용하고 context를 직접 변경하지 않으므로, 당시 16k base config는 별도 보존이 필요한 재현성 항목입니다.

## 최종 내부 개발 버전 v58

- 생성 코드: 01_workspace/train_v50.py
- 설정 산출물: 03_submission/model/config.json, recipe.yaml
- 양자화: AWQ W4A16 asymmetric
- context length: 16,384
- RoPE factor: 2.0
- rope theta: 10,000
- KV cache: FP8 설정
- batch size: 1
- 실제 코드의 calibration max sequence length: 768

v58은 내부 개발이 완료된 최종 버전입니다. 다만 최고 평가 점수 0.6153749319는 제출 파일 submit_v37_absolute.zip에 연결된 기록이므로, 별도의 제출 증거 없이 v58의 점수로 직접 귀속하지 않습니다.

## Optimization Roadmap

### Phase 1 — 기준선과 양자화 탐색

- FP8 W8A8 기준선 구축
- GPTQ W4A16 및 activation order 비교
- MANTA 중심 calibration과 multi-dataset calibration 비교

### Phase 2 — 데이터 구성과 정밀도 방어

- MANTA·GSM8K·KMMLU·MBPP 조합 실험
- embed_tokens와 lm_head 보호
- FP8 precision-defense 실험
- calibration sample 수와 sequence length 조정

### Phase 3 — 최고 평가 제출

- MANTA 128, GSM8K 128, KMMLU 256으로 총 512개 구성
- FP8 W8A8
- submit_v37_absolute.zip 생성
- Public Score 0.6153749319, 평가 시간 9분 45초

### Phase 4 — 내부 v58까지 확장

- AWQ와 Duo Scaling 적용
- 16k context와 RoPE 설정 조정
- FP8 KV cache 구성
- batch size 1로 calibration 메모리 피크 억제
- 03_submission/model에 최종 설정 산출물 보존

## 주요 실패와 학습

- 일부 GPTQ W4A16 설정에서 수학·논리 성능 하락
- 초기 AWQ 실험에서 기존 최고점보다 낮은 평가 결과 관측
- max_position_embeddings와 입력 길이 충돌로 decoder prompt 오류 발생
- quantization kernel과 하드웨어 설정 불일치로 engine initialization 오류 발생
- calibration sample과 sequence length 증가 시 메모리 부족 발생

실험 결과는 특정 양자화 알고리즘 하나가 항상 우월하기보다, calibration data·context·kernel·메모리 예산의 상호작용이 중요하다는 점을 보여줬습니다.

## Technical Stack

| Category | Specification |
|---|---|
| Base Model | EXAONE 4.0 1.2B Instruct |
| Quantization | FP8 W8A8, GPTQ W4A16, AWQ W4A16 |
| Context | 최고 제출 기록 16k, 내부 v58 16k |
| Data | MANTA, GSM8K, KMMLU, MBPP |
| Runtime | vLLM, llmcompressor, transformers |
| Hardware | L4 GPU 환경 |

## Evidence Boundary

- 모델 weight와 제출 ZIP은 GitHub 용량 제한으로 제외했습니다.
- 최고 평가 결과는 제출 기록으로 확인했습니다.
- 중간 버전 점수는 milestones.md의 실험 기록이며, 원본 캡처가 없는 값은 별도 증거 보강이 필요합니다.
- 원본 결과 파일이 없는 latency 비교는 확정 성과로 사용하지 않습니다.
