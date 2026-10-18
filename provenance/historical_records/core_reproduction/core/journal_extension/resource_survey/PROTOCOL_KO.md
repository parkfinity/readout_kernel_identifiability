# 회로 계열과 readout별 추가 실행 자원 조사

기존 full 실험 뒤의 결정 실험이다. 조사 범위는 이전에 기록한 120개 슬롯을 유지한다. 레이어 인자가 없는 중복 18개는 실행하지 않고 명시한다. 기존 13모듈 중 기본 생성 규칙으로 구성하지 않은 3모듈의 제외 사유도 유지한다. 이는 사전등록 연구나 문헌 전체 발생률 조사가 아니다.

## 고정 비교와 검증

기존 원본 바이트와 어댑터를 core에 복사해 이전 배포판은 변경하지 않았다. Qiskit level 3 컴파일 뒤 readout별 공통 등거리 suffix 제거, 기존 restricted boundary 운반, 같은 축 재융합을 기준선으로 사용한다. 새 suffix 기준선은 전체 suffix가 입력 독립인 등거리 연산들로 구성된 경우를 처리한다. 또는 후보와 겹치는 연산들이 모두 후보 support 안에 머무는 공통 등거리이면, 겹치지 않는 연산을 교환할 수 있다. 이 support 제한은 얽힘 게이트 뒤에 처음에는 겹치지 않았던 데이터 게이트가 영향을 미치는 경우를 막는다. 2큐비트 전체 Pauli 등방 readout의 일반 공통 출력 회전도 기준선에서 처리한다. 이 규칙을 완전한 기호 최적화라고 주장하지 않는다.

그 후 기존 metric transport를 추가해 남은 기호 수, 융합 연산 수, 학습 가능한 gate-angle 작업 수를 비교한다. 모든 미지원 생성·표현과 검증 실패를 공개한다. 검증 실패는 양성/음성 결과로 합치지 않는다.

새 gate-angle 공유 미분기는 각 융합 Pauli 회전의 특징값에 두 shift를 적용하고, RBF 및 목적함수에는 chain rule을 적용한다. 한 각도 안의 여러 기호는 미분 특징을 재사용하고, 여러 게이트에 공유되는 기호는 모든 occurrence 기여를 합산한다. RBF 커널 자체에 two-shift rule을 적용하지 않는다.

기본 constructor 인자는 고정 소스의 기본값이며 random 회로 seed는 0이다. 실제 kwargs, constructor signature, get_params를 슬롯별로 저장한다. Readout은 local XYZ, local Z, Pauli weight 1 또는 2의 전체 nonidentity 관측량이며 metric은 모두 단위 가중치다. 큐비트 2/4, 층 1/2를 사용한다.

Probe seed 812와 1907에서 각각 입력 5개와 0이 아닌 초기 파라미터를 사용한다. 원본 Qiskit 특징과 compiler IR 특징, 세 arm의 kernel, angle-shared와 analytic gradient, compiler-IR 중앙 유한차분, 두 update 경로와 ridge 예측을 검사한다. 원본 source의 모든 perturbed parameter에 유한차분을 재실행하는 실험은 아니다. 이 작은 검사는 이론적 soundness 증명을 대체하지 않는다.

## 결과 사용 기준

최종 융합 연산이나 angle 작업이 실제로 줄어드는 슬롯을 후속 반복 timing 후보로 분류한다. 기호 수만 감소하면 parameter-wise backend 효과로 분리한다. 물리 게이트 수, QPU shots, 고정 정확도 shot 수는 이 simulator 구조 조사로 측정하지 않는다. 인증 단계의 단발 진단 시간은 속도 향상의 근거로 사용하지 않는다. 여러 readout, 크기, 같은 계열의 깊이 변화는 독립적인 회로 계열이 아니다.

## 서버 실행

Python 3.8 환경에서는 core/requirements_py38.txt의 고정 버전을 사용한다. run.sh는 단위 테스트 뒤 전체 조사를 실행하고 results에 JSON 및 수치 배열과 manifest를 저장한다. 이전 서버 환경을 재사용할 때에도 패키지 버전을 먼저 확인한다.

```bash
cd /workspace
python3 -m zipfile -e cross_family_resource_v1.zip .
cd cross_family_resource_v1
source /workspace/.venv_external_pqk/bin/activate
python3 -m pip install -r core/requirements_py38.txt
bash run.sh
```

로컬 Mac 실행과 Linux 서버 재현은 별도 run으로 보존한다. 로컬 결과를 서버 실행 완료라고 부르지 않는다.
