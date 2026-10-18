# 세 회로 계열 확장 실험 실행 안내

YZ-CX, Hubregtsen, Kyriienko의 10개 원본 회로 구성을 3개 readout으로 평가하는 30개 설정 실험이다. 이전 ChebyshevRx·ParamZ 24개 설정과 겹치지 않는다. 원본 모델 코드, 고정 입력 IR, 단위 테스트, 로컬 결과와 인증 재생 코드를 함께 제공한다. 전체 102개 재실행이나 QPU 실험은 아니다.

## 서버에서 재현

ZIP을 `/workspace`에 올리고 아래 명령의 내용만 복사한다. Markdown의 백틱 줄은 터미널에 붙이지 않는다. 별도 `zip` 프로그램은 필요 없다.

```bash
cd /workspace
python3 -m zipfile -e certified_dimension_extension_v1_bundle.zip .
cd certified_dimension_extension_v1
python3 -m venv /workspace/.venv_dimension_extension
source /workspace/.venv_dimension_extension/bin/activate
python3 -m pip install -r requirements_py38.txt
bash run.sh smoke
bash run.sh replay
bash run.sh source-check
bash run.sh closure
bash run.sh nonlinear
python3 -m zipfile -c server_dimension_extension_audits.zip server_replay.json server_source_check.json server_closure.json server_nonlinear_hubregtsen.json server_nonlinear_source_check.json
```

먼저 `replay`를 권한다. 전달된 정확 항등식, 원래 입력점의 ACTIVE 증거, 소행렬식과 자원 arm을 높은 정밀도로 다시 계산한다. `source-check`는 번들에 포함한 해시 고정 라이브러리 소스와 비교한다. 전체 sQUlearn 설치 환경이 아니라 기존 연구에서 사용한 pinned-source host이다. `closure`는 기존 구조 기준선의 비활성 인증과 각도 계수 랭크 상한을 함께 적용하는 별도 사후 분석이다.

새 실행 결과를 처음부터 생성하려면 다음을 사용한다.

```bash
bash run.sh full
```

`full`은 새 UTC 시각의 결과 폴더를 만든다. 20초 기호 계산 예산 때문에 CPU 속도에 따라 기호 판정 완료 개수는 달라질 수 있다. 이것을 서로 다른 수학적 결론으로 해석하지 않는다. 동일한 저장 증거의 재생과 새 예산 제한 실행은 다른 검사다. 활성 증거점과 입력은 고정되어 있다. 30개 설정이 모두 끝나면 `summary.json`과 `manifest.json`이 생성된다.

## 파일과 검증 범위

- `PROTOCOL_KO.md`: 실행 전 고정한 범위와 실패 처리, 자원 기준선.
- `experiment.py`: 정확 항등식, 구간 미분, 소행렬식, global-isometry 및 강화 commutant 비교.
- `exact_engine.py`: 정수·반정수 계수만 지원하며 근삿값을 π로 바꾸지 않는다.
- `interval_engine.py`: 실제 θ·x 공유를 지킨 독립적인 statevector 및 미분 계산.
- `source_core/`: 고정한 외부 라이브러리 소스와 기존 수치 평가기.
- `structural_closure.py`: 기호 전개가 예산을 넘은 경우에도 기존 구조 증거가 있으면 별도 표기.
- `RESULTS_KO.md`: 로컬 실행 결과와 해석.

mpmath 구간 계산은 실험적 소프트웨어이고 증명 보조기로 검증된 arithmetic kernel은 아니다. Fourier 전개의 비영 계수나 작은 부동소수 미분은 ACTIVE/INACTIVE 수락 기준이 아니다. Pauli 회전 작업과 두 점 shift 묶음은 분석적 회계이며 물리 기본 게이트, 실제 shot 실행, 동일 정확도 shot 절감 또는 end-to-end 시간 절감 실측과 다르다.

로컬 환경은 Python 3.9.6, NumPy 1.24.4, SymPy 1.14.0, mpmath 1.3.0, Qiskit 1.2.4다. requirements의 SymPy 1.13.3은 Python 3.8 서버 호환용이며 실제 Linux 서버 검증은 아직 별도다. 원 실행 당시 스냅샷과 이후 추가한 감사·배포 스크립트는 RELEASE 해시에서 구분된다.

첫 로컬 실행은 21개 설정을 저장한 뒤 기호 계산 예산 초과 예외와 타이머 종료가 겹쳐 중단되었다. 예외 처리 전에 타이머를 해제하도록 수정하고 나머지 설정을 재개했다. 저장된 21개 결과는 변경하지 않았으며 `resume.json`에 해시가 있다. 계산 예산, 범위, 증거점 및 판정 규칙은 바꾸지 않았다. 원본 코드와 재개 코드를 모두 보존한다.

Hubregtsen 4큐비트·1층 local Z의 차원 간극은 별도 사후 분석으로 조사했다. `nonlinear`는 네 특징의 정확 지수다항식과 두 비선형 좌표 표현을 검증하고, 원본 소스 1,000개 입력 쌍 및 100개 미분점에 대조한다. 이 검증은 원래 예산 제한 판정 결과를 덮어쓰지 않는다.
