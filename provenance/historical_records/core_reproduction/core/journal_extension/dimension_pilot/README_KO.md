# 인증 차원 파일럿 실행 패키지

이 패키지는 24개 제한 설정의 실제 실행 코드와 고정 입력, 로컬 결과를 포함한다.
102개 전체 설정용 완전 판정 엔진은 아니다. 기존 환경이나 원고를 덮어쓰지 않는다.

## 확인된 결과

- 두 회로 계열 × 두 qubit 수 × 두 layer 수 × 세 readout = 24개 설정.
- 기호 162개: 정확 비활성 30, 실제 입력 ACTIVE 증거 132, 미결정 0.
- 24개 모두 차원 상한과 하한 일치. 22개 비영 minor, 2개 상수 커널.
- 기존 이 부분집합의 inactive 18개 회수, 새 inactive 12개.
- 새 12개는 ParamZ/local-Z의 상수 출력 또는 최종 Z 회전 불변성이다. 강한 commutant 기준선 이후 새로운 자원 이득이라는 뜻이 아니다.
- 로컬 원 실행 82.65초. 학습 속도 절감 수치가 아니라 인증 생성 비용이다.

## 서버에서 재생

ZIP을 `/workspace`에 올린 후 아래 코드 블록의 명령만 복사한다. 백틱은 복사하지 않는다.

```bash
cd /workspace
python3 -m zipfile -e certified_dimension_pilot_v1_bundle.zip .
cd certified_dimension_pilot_v1
python3 -m venv /workspace/.venv_certified_dimension
source /workspace/.venv_certified_dimension/bin/activate
python3 -m pip install -r requirements_py38.txt
bash run.sh smoke
bash run.sh replay
```

`replay`는 포함된 원래 24개 인증서를 더 높은 정밀도로 확인하고 `server_replay.json`을 만든다.
이를 먼저 실행하면 서버와 로컬의 탐색 차이 없이 증거 자체를 비교할 수 있다.
새로 전 과정을 생성하려면 이어서 실행한다.

```bash
bash run.sh pilot
```

새 결과는 `results/pilot_<UTC timestamp>/`에 생성되며, 원래 결과를 덮어쓰지 않는다.
Qiskit과 데이터셋은 smoke/replay/pilot에 필요 없다. `source_crosscheck.py`만 기존 프로젝트의
고정 sQUlearn source host와 Qiskit에 의존하는 별도 개발용 진단이다.

원 로컬 실행은 Python 3.9.6, SymPy 1.14.0, NumPy 1.24.4, mpmath 1.3.0이었다.
requirements는 기존 Python 3.8 서버용 잠금이다. 실제 서버 결과는 아직 받지 않았으므로 서버 통과를 주장하지 않는다.
Python 3.8 문법 검사와 패키지 추출 smoke/replay 검증은 로컬에서 수행한다.

## 결과 압축

시스템 `zip` 명령이 없어도 다음 명령을 사용할 수 있다.

```bash
python3 -m zipfile -c certified_dimension_server_results.zip server_replay.json results
```

`results`에는 포함된 원래 로컬 run도 있으므로 어느 run이 서버에서 새로 생성됐는지 환경 기록으로 구분한다.

## 해석 제한

정확 영항등식과 interval 비영 증거를 분리한다. 수치 SVD·작은 gradient·무작위 all-zero는 인증으로 사용하지 않는다.
24개는 선택한 제한 범위다. 일반 prevalence, 102개 완전 판정, QPU gate/shot 절감 또는 신규성 확정으로 해석하지 않는다.
mpmath interval 연산과 구현은 proof assistant로 검증된 소프트웨어가 아니다.
원본 source↔IR은 별도의 수치 진단도 통과했지만 전체 컴파일러의 형식 검증을 뜻하지 않는다.

파일별 역할: `exact_engine.py`는 정확 계수 항등식, `interval_engine.py`는 원래 입력 상태·미분,
`run_pilot.py`는 실행/증거 재생, `test_engine.py`는 16개 테스트다.
원 run의 `source_snapshot`은 당시 실행 코드이고, 현재 `run_pilot.py`에는 배포용 입력 로딩과 추가 재생 검사가 더해져 있다.
원 run의 manifest는 변경하지 않는다. 배포 전체 해시는 `RELEASE.json`에서 별도로 확인한다.
