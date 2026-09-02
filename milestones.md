# Submission & Performance Tracking

EXAONE 4.0 1.2B 경량화 실험의 제출 기록과 내부 개발 버전을 구분해 관리합니다.

## 증거 등급

- **제출 검증**: 제출 화면·파일명·점수·시간을 직접 확인
- **내부 기록**: 프로젝트 진행 중 기록한 점수이며 추가 캡처가 필요
- **코드 검증**: 구현과 설정 파일은 존재하지만 별도 제출 점수가 연결되지 않음

## 주요 기록

| 버전 | 일시 | 주요 설정 | 점수 | 평가 시간 | 증거 상태 | 비고 |
|---|---|---|---:|---:|---|---|
| v9 | 02-05 | FP8 W8A8 정밀 교정 | 0.49491 | 13분 11초 | 내부 기록 | 초기 안정권 |
| v11.6 | 02-06 | GPTQ W4A16 Static | 0.59491 | 10분 28초 | 내부 기록 | 첫 급성장 |
| Early Peak | 02-03 | GPTQ W4A16, S512/L512 | 0.60918 | 10분 14초 | 내부 기록 | 초기 고점 |
| v24 | 02-13 | FP8 W8A8 Precision Defense | 0.60074 | 10분 07초 | 내부 기록 | FP8 고점 |
| **v37 제출명** | **02-17** | **FP8 W8A8, calibration 512, L1024, 제출 기록 16k** | **0.6153749319** | **9분 45초** | **제출 검증** | submit_v37_absolute.zip |
| v40 계열 | 02-22 | AWQ, S128×3, L768, 16k | 0.48547 | 13분 58초 | 내부 기록 | 초기 AWQ 하락 관측 |
| FP8 Dynamic 16k | 02-24 | FP8 Dynamic, 16k | 0.52174 | 12분 43초 | 내부 기록 | 후기 비교 실험 |
| **v58 내부 최종** | 최종 | **AWQ W4A16, 16k, S96×3, L768, batch 1** | — | — | **코드 검증** | train_v50.py 및 03_submission/model |

## 버전명 해석

- v37은 최고 평가 제출 파일명에 남은 버전입니다.
- v58은 이후 내부 개발을 계속한 최종 버전입니다.
- v58이 v37보다 높은 번호인 것은 정상이며, 최고 제출 기록과 내부 최종 개발 상태를 서로 다른 열로 관리합니다.
- 일부 파일명과 코드 내부 출력 버전은 다릅니다. 파일명은 workspace snapshot이고 코드 내부 버전은 당시 실험 관리 번호이므로, 파일명만으로 실행 순서를 추정하지 않습니다.

## v37 최고 평가 제출

- 코드: 01_workspace/final.py
- 출력: submit_v37_absolute.zip
- 방식: FP8 W8A8
- calibration: MANTA 128 + GSM8K 128 + KMMLU 256
- max sequence length: 1024
- 제출 기록 context: 16,384
- Public Score: 0.6153749319
- 평가 시간: 9분 45초

final.py는 외부 base_model을 불러오고 context를 직접 설정하지 않습니다. 실행 당시 16k base config 또는 선행 notebook 셀은 추가 보존이 필요합니다.

## v58 내부 최종 개발

- 코드: 01_workspace/train_v50.py
- 출력 설정: 03_submission/model/config.json, recipe.yaml
- 방식: AWQ W4A16 asymmetric, Duo Scaling
- calibration: MANTA 96 + GSM8K 96 + KMMLU 96, 총 288개
- max sequence length: 실제 코드 기준 768
- context: 16,384
- RoPE factor: 2.0
- rope theta: 10,000
- KV cache: FP8 설정
- batch size: 1

v58은 완성된 내부 개발 버전입니다. 별도 제출 결과가 추가되면 해당 점수와 캡처를 이 표에 연결합니다.

## 기술적 해석

### 양자화 방식

- GPTQ W4A16은 일부 초기 제출에서 높은 기록을 냈지만 설정에 따라 engine initialization 오류와 품질 변동이 있었습니다.
- FP8 W8A8 기반 v37 제출이 현재 확인된 최고 평가 기록입니다.
- AWQ는 초기 실험에서 하락도 있었지만 내부 개발을 v58까지 이어가며 16k·RoPE·KV cache·메모리 설정을 통합한 최종 산출물을 완성했습니다.

### 실패 사례

- Decoder prompt empty: context 설정과 실제 입력 길이 충돌
- Engine initialization failure: quantization kernel과 하드웨어 설정 불일치
- Killed/OOM: calibration sample·sequence length·batch size에 따른 메모리 피크

## 남은 증거 보강

- v9·v11.6·Early Peak·v24·v40 계열·FP8 Dynamic의 제출 캡처 또는 원본 평가 로그
- v58 별도 제출 여부와 평가 결과
- v37 실행 당시 사용한 실제 16k base config
- README에 과거 기재됐던 latency 10.7ms/t와 8.4ms/t의 원본 benchmark 결과
