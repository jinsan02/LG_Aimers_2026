# EXAONE 4.0 1.2B 경량화 및 추론 최적화

Hanshin University Computer Engineering | Roh Jin-san (GPA 3.7/4.5)

LG AI Research의 EXAONE 4.0 1.2B를 제한된 L4 GPU 환경에서 경량화하면서, 양자화 방식·캘리브레이션 데이터·문맥 길이가 정확도와 추론 비용에 미치는 영향을 반복 실험한 프로젝트입니다.

## 핵심 결과

| 구분 | 결과 | 근거 |
|---|---|---|
| 최고 평가 제출 | submit_v37_absolute.zip, Public Score **0.6153749319**, 평가 시간 **9분 45초** | 제출 기록 및 01_workspace/final.py |
| 최고 제출 설정 | FP8 W8A8, calibration 512개, max sequence length 1024 | 01_workspace/final.py |
| 최종 내부 개발 버전 | **v58**, AWQ W4A16·16k context·RoPE 조정·FP8 KV cache | 01_workspace/train_v50.py, 03_submission/model/ |

> 제출 파일명에 남은 v37과 내부 최종 버전 v58은 서로 다른 관리 시점의 버전명입니다. 최고 평가 기록과 최종 내부 개발 산출물을 혼동하지 않도록 분리해 기술합니다.

## final.py의 역할

01_workspace/final.py는 저장소에서 가장 높은 평가 기록을 낸 submit_v37_absolute.zip 생성 스크립트입니다. 이름의 final은 최신 내부 버전 번호가 아니라 **최고 평가 제출용 빌드 스크립트**라는 의미입니다.

- MANTA 128개
- GSM8K 128개
- KMMLU 256개
- 총 calibration sample 512개
- calibration max sequence length 1024
- weight: 8-bit FLOAT, static
- input activation: 8-bit FLOAT, dynamic
- embed_tokens, lm_head 제외

제출 화면에는 16,384 context가 기록돼 있습니다. 다만 final.py는 context 값을 직접 변경하지 않고 외부 /content/drive/MyDrive/comp/base_model 상태를 사용하므로, 정확한 재현에는 실행 당시 base config 보존이 추가로 필요합니다.

## 내부 최종 버전 v58

01_workspace/train_v50.py는 내부 개발이 v58까지 진행된 뒤 사용한 AWQ 경량화 스크립트입니다.

- AWQ W4A16 asymmetric
- 16,384 context
- RoPE factor 2.0
- rope theta 10,000
- FP8 KV cache 설정
- batch size 1
- 실제 코드 기준 calibration max sequence length 768

03_submission/model/config.json과 recipe.yaml은 v58 계열 설정을 보존합니다. GitHub 용량 제한 때문에 모델 weight와 제출 ZIP은 저장소에 포함하지 않습니다.

## 실험 과정

1. FP8 W8A8 기준선 구축
2. GPTQ W4A16 및 activation order 실험
3. MANTA·GSM8K·KMMLU·MBPP calibration 구성 비교
4. FP8 precision-defense 실험
5. 16k context 확장과 메모리 병목 대응
6. AWQ·Duo Scaling 전환
7. batch size 조정과 Diet Calibration으로 메모리 피크 억제

실험은 항상 점수가 개선된 것은 아닙니다. 특히 일부 AWQ 초기 실험에서는 정확도 하락이 관측됐고, 이를 통해 양자화 알고리즘 자체보다 calibration 구성과 context·kernel 설정의 상호작용이 중요하다는 점을 확인했습니다.

## 개인 역할

- FP8·GPTQ·AWQ 양자화 방식 비교
- calibration dataset 구성과 sampling 전략 설계
- 8k→16k context 확장 및 RoPE 설정 조정
- 메모리 부족·vLLM 초기화 오류·prompt 오류 해결
- 제출 모델 패키징과 benchmark pipeline 구성

## 재현성과 한계

- 저장소는 32개의 실험·benchmark 스크립트를 보존하지만, 초기 개발 이력이 한 번의 공개 커밋으로 정리돼 Git commit만으로 시간순 실행을 복원할 수 없습니다.
- 파일명과 내부 실험 버전이 다른 경우가 있어 milestones.md에서 대응 관계를 설명합니다.
- 모델 weight, ZIP, 일부 원본 평가 로그는 GitHub 용량 제한으로 제외했습니다.
- 최고 점수 0.6153749319와 9분 45초는 제출 기록으로 확인했으며, 다른 중간 점수는 milestones.md의 실험 기록으로 구분합니다.
- README에 원본 결과 파일이 없는 latency 수치는 사용하지 않습니다.

## Repository Map

- 01_workspace/final.py: 최고 평가 제출 v37 빌드
- 01_workspace/train_v50.py: 내부 최종 v58 AWQ 빌드
- 01_workspace/benchmark*.py: 정확도·latency 평가 도구
- 03_submission/model/: v58 설정 및 tokenizer 산출물
- history.md: 개발 단계와 기술 의사결정
- milestones.md: 제출·실험 버전 기록
