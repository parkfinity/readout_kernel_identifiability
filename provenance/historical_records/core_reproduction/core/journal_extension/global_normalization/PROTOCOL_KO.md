# 전역 Pauli 정규화와 재매개변수화 결정 실험

이 실험은 대표 ChebyshevRx 사례가 기존의 전역 단순화 이후에도 새로운 실행 자원 절감을 주는지 점검한다. 조사 범위를 추가로 늘리거나 18,900 fits를 반복하는 실험이 아니다. 기존 12개 설정과 local XYZ readout을 고정한다. 이 패키지는 QuCLEAR나 Paulihedral 전체 시스템의 구현 또는 head-to-head 비교가 아니다.

## 고정 비교

ChebyshevRx, ChebyshevPQC, YZ-CX의 2·4큐비트, 1·2층 총 12개 설정을 사용한다. 소스는 동봉 manifest의 sQUlearn commit 0f6cb0982fdbe80dcd0ea6cb0acd68c59e362de4에 고정한다. 전체 sQUlearn 실행 환경이 아니라 원본 소스와 명시적인 호환 host를 사용한다.

compiler, old boundary, old transport, global unitary, global unitary 후 common-isometry 제거, old boundary 후 global, old transport 후 global의 7개 표현을 저장한다. 전역 결합은 부호 있는 Clifford 운반과 교환 가능한 구간의 동일 Pauli 합산만 사용한다. 최종 Clifford는 항등인 경우에만 생략하며 측정 변경으로 흡수하지 않는다. common-isometry 제거는 기존 readout criterion을 재사용한다. 이를 독립적인 기존 알고리즘과의 우월성 실험으로 해석하지 않는다.

## 검증

- 설정마다 20개 probe, 각 5개 입력. 입력 범위 [-0.8,0.8], 파라미터 범위 [-1,1], seed 331.
- Qiskit 원본 unitary와 별도 dense matrix 곱, 원본 feature, 모든 arm의 커널·gradient 비교. 일반 오차 한계 1e-9.
- 첫 probe의 모든 파라미터에 central finite difference, step 1e-6, 오차 한계 1e-6.
- 별도 2큐비트 해석식: seed 90210, 1,000개 입력쌍, 입력 [-0.95,0.95], 파라미터 [-2,2]. 오차 한계 1e-10.
- 12개 입력의 고정 학습 diagnostic: 20 alignment-ascent steps, eta 0.1. beta=A theta에 대해 AAᵀ metric을 쓰는 경로와 단순 beta gradient 경로를 구분한다.
- 독립 checker는 표준 라이브러리의 Pauli 곱셈으로 부호·fusion·삭제 기록을 검사한다. source adapter나 정리 자체의 형식 증명은 아니다. 변조된 sign/output은 거절해야 한다.

angle coefficient rank는 동일하게 정리된 feature expression을 기저로 한 선형 계수 rank다. 임의 함수 관계를 해결하지 않으며 kernel의 최소 차원으로 보고하지 않는다. Pauli rotation task는 native 물리 게이트 수가 아니다. 새 속도 향상·고정 정확도 shots·QPU 절감을 주장하지 않는다.

## 서버 실행

먼저 `global_pauli_reparameterization_v1_code.zip`을 서버 `/workspace`에 업로드한다. 아래 코드블록 안의 명령만 복사한다. 시작과 끝의 백틱 줄은 복사하지 않는다.

```bash
cd /workspace &&
source /workspace/.venv_external_pqk/bin/activate &&
python3 -m zipfile -e global_pauli_reparameterization_v1_code.zip global_pauli_server_run &&
cd global_pauli_server_run/global_pauli_reparameterization_v1 &&
python3 -m pip install -r core/requirements_py38.txt &&
bash run.sh &&
python3 -m zipfile -c /workspace/global_pauli_server_results.zip results
```

`results/global_.../summary.json`의 passed와 checker.passed를 확인한다. 위 명령은 기존 실험을 변경하지 않는 별도 폴더를 사용한다. ZIP을 전달하면 로컬 결과와 비교할 수 있다. checker만 다시 돌리려면 `python3 checker.py results/global_실제폴더명`을 사용한다.

## 범위와 다음 판단

성공은 검증 통과를 뜻하며 연구 가설의 우월성을 뜻하지 않는다. 추가 angle-task 절감이 0이면 그대로 보고한다. 전역 정규화가 양쪽 arm에 공통으로 주는 절감은 제안 인증기의 효과에 합산하지 않는다. 이 진단 이후에도 새 사례가 없으면 대규모 timing을 자동 실행하지 않는다. 다음 후보는 제한 클래스의 완전성/한계 분석, certificate-only 규모 확장과 별도 checker 비용 측정이다. 이 패키지에 해당 대규모 실험이 완료됐다고 표시하지 않는다.
