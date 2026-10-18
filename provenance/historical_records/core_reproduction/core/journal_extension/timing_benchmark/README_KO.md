# 내부 인증 추가 효용 실험

이번 실험은 컴파일·국소 말단 제거·단순 boundary transport·같은 축 회전 합치기 이후에도 **후속 데이터 게이트를 통과하는 인증**이 추가로 유용한지 확인합니다. 서버 원자료에서 동결한 입력이 포함되어 있어 원본 영상이나 이전 실험 폴더는 필요하지 않습니다. 새 virtualenv를 쓰는 것이 기존 실험 환경을 보존하는 방법입니다.

## 서버 실행

`interior_incremental_v1.zip`을 `/workspace/`에 업로드한 뒤 아래 명령만 복사합니다. 코드 블록의 시작/끝 백틱은 복사하지 않습니다. 외부 `zip`/`unzip` 프로그램은 필요하지 않습니다.

```bash
cd /workspace &&
test -f interior_incremental_v1.zip &&
test ! -e interior_incremental_v1 &&
python3 -m zipfile -e interior_incremental_v1.zip . &&
cd interior_incremental_v1 &&
python3 -m venv /workspace/.venv_interior_incremental &&
source /workspace/.venv_interior_incremental/bin/activate &&
python3 -m pip install -r requirements_py38.txt &&
bash run.sh smoke
```

같은 이름의 폴더가 이미 있으면 위 명령은 덮어쓰지 않고 멈춥니다. 기존 결과를 지우지 말고 기존 폴더인지 먼저 확인합니다. 패키지·입력 무결성을 자동 검사하고, Qiskit 버전이 다르면 실행을 중단합니다.

Smoke 마지막에 `COMPLETE_AND_AUDITED`가 표시되고 `passed: true`인지 확인한 뒤 진행합니다.

```bash
cd /workspace/interior_incremental_v1 &&
source /workspace/.venv_interior_incremental/bin/activate &&
bash run.sh full
```

중간 규모 서버 점검을 원하면 full 전에 `bash run.sh pilot`을 실행합니다. 로컬 pilot은 이미 확인했지만 Python 3.8 서버 결과는 별도 검증 대상입니다. 실행별 타임스탬프 폴더를 만들므로 재실행해도 이전 결과를 덮어쓰지 않습니다. 계산 시간은 서버에 따라 달라지며 수만 회의 fit과 전체 감사가 포함됩니다. 다른 무거운 작업과 동시에 실행하지 않는 편이 시간 비교에 적합합니다.

실행이 끝나면 결과 폴더와 아래 ZIP을 자동 생성합니다.

```text
/workspace/interior_incremental_v1/results/full_날짜시간.zip
```

마지막 출력의 ZIP을 그대로 전달하면 됩니다. 숫자 결과, 각 단계 timing, 전체 학습 trace, Gram 배열, certificate, source snapshot, 환경, hash manifest와 audit가 들어 있습니다. `summary.json`과 `audit.json` 모두 passed=true여야 완료입니다. 실패 ZIP도 분석용으로 보존되며 완료 결과로 취급하지 않습니다. 중단 시 콘솔 오류와 해당 결과 ZIP을 전달합니다.

## 무엇을 비교하는가

- 주 대상: ChebyshevRx 2·4 qubits, 2 layers. 나머지 10개 회로 설정은 대조군입니다.
- 주 비교: `boundary` 대 `transport`. 전자는 단순 말단·affine bias 등을 이미 제거합니다. 단순 `commutant`만 기준선으로 쓰지 않습니다.
- 같은 축 연속 회전은 양 군 모두 다시 합칩니다. IR을 잘게 나눈 것만으로 생기는 인위적 gate 비용 차이를 줄였습니다.
- Serial, batched, adjoint를 별도로 측정합니다. 호출 수·미분 항·gate update를 구분합니다.
- 세 데이터셋×5 seeds의 동결 입력, 5회 paired 반복입니다. 15개 독립 문제라고 표현하지 않습니다.
- 추가 인증 비용을 포함한 cold total과 실제 H=1·2·5·10·20 재사용 시간을 기록합니다. 재사용 실험은 seed42 부분집합입니다.
- 모든 비교군과 성능이 나빠지는 결과도 보존합니다. Simulator 결과이며 QCAS 일반 가속이나 QPU saving 실험이 아닙니다.

전체 설계와 pilot 이후 기준선을 강화한 기록은 `PROTOCOL.md`, 로컬 검증 범위는 `LOCAL_VALIDATION_KO.md`에 있습니다. 기존 라이브러리 소스 attribution과 라이선스는 `vendor/` 및 `source_manifest.json`에 보존했습니다.
