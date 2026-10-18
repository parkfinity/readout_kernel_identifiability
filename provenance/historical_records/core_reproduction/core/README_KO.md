# 재현 번들 v1: Fourier–Gram 30개 설정 + Kyriienko n = 2, 4, 6, 8

이 번들로 가장 최근의 인증 계산 두 개를 서버에서 다시 실행하고, 저장된 참조 결과와 정확히 같은지 기계적으로 판정할 수 있다.

| 계산 | 원본 폴더 | 참조 결과 | 내용 |
| --- | --- | --- | --- |
| (A) Fourier–Gram 폐쇄 | `fourier_gram_identifiability_v1/experiment` | `20261002T044925Z` | 동결 IR 30개의 정확 랭크 인증이다. Kyriienko 구간 [4,5]·[10,11]·[10,11]이 4·10·10으로 닫힌다. 기호별 판정은 ACTIVE 195, INACTIVE 57, UNDECIDED 0이다. Qiskit은 쓰지 않는다. |
| (B) Kyriienko 스케일링 | `kyriienko_scaling_v1` | `20261003T121202Z` | Qiskit 1.2.4 transpile로 IR을 만들고 n = 2, 4, 6, 8 × readout 3개, 모두 12개 설정을 인증한다. XYZ와 weight≤2는 r = 3n−2, local Z는 r = 2n이다(n = 2 weight≤2는 0). |

## 1. 구성

```
reproduction_bundle_v1/
  run_all.sh               단일 진입점 (스레드 고정, 버전 점검, 테스트, 두 계산, 비교, 압축)
  compare_results.py       새 실행과 참조 결과 비교 -> comparison_verdict.json
  tools/repro_helper.py    버전 점검(env), 번들 무결성(integrity), 단계 실행·시간·메모리 기록(run),
                           Python 3.8용 math.gcd 호환 실행(pyexec)
  requirements_server.txt  numpy 1.24.4, sympy 1.13.3, mpmath 1.3.0, qiskit 1.2.4, scipy 1.10.1
  BUNDLE_MANIFEST.json     모든 파일의 SHA256과 원본 경로. 복사본은 원본과 모두 바이트 단위로 같다
  journal_extension/       원본 트리에서 필요한 부분만 같은 상대 경로로 복사한 것 (스크립트를 수정 없이 실행하기 위함)
    fourier_gram_identifiability_v1/experiment/   fourier_gram.py, run_experiment.py, test_fourier_gram.py, copied_from_extension_v1/
    kyriienko_scaling_v1/                         kyriienko_scaling.py, test_kyriienko_scaling.py, core_copy/ (vendored sQUlearn 0f6cb098 포함)
    cross_family_resource_v1/core/                core_copy의 원본 (provenance 해시 비교용)
    certified_dimension_extension_v1/             frozen_inputs/ (IR 30개 + scope.json), 원 실행 결과 JSON 30개 + manifest,
                                                  structural_closure.json, nonlinear_hubregtsen.json, exact_engine.py, interval_engine.py
    manuscript_census_reference/extension_results.tex          (A)가 원고 표를 읽는다
    manuscript_theory_reference/{family_extension,fourier_gram,dimension_theory}.tex   (B)가 해시만 기록한다
  reference_results/       fourier_gram/20261002T044925Z/, kyriienko_scaling/20261003T121202Z/ (원본 결과 폴더 그대로)
  local_validation/        로컬 검증 실행 기록 (아래 6절): 환경 3개의 판정·로그, 음성 대조, gcd shim 없는 Py3.8 실패 로그, 메모리 측정
```

`journal_extension/` 안의 코드는 원본과 SHA256이 같다. 비교 단계에서는 새 실행의 `provenance.json`에 기록된 코드 해시, 동결 입력 해시, 원고 해시가 참조 실행과 같은지도 확인한다.

## 2. 서버 실행 방법

```bash
# (로컬) 전송
scp reproduction_bundle_v1.zip <user>@<server>:~/

# (서버) 압축 해제 전에 zip SHA256을 전달받은 값과 대조한다
cd ~ && sha256sum reproduction_bundle_v1.zip
unzip -q reproduction_bundle_v1.zip && cd reproduction_bundle_v1

# 기존 환경(Python 3.8.10, NumPy 1.24.4, Qiskit 1.2.4, SymPy 1.13.3)이 있으면 그 인터프리터를 지정한다
PY=/path/to/venv/bin/python bash run_all.sh

# 환경이 없으면 새 venv를 만든다 (PyPI 접근 필요)
python3 -m venv ~/venv_repro_v1
~/venv_repro_v1/bin/pip install -r requirements_server.txt
PY=~/venv_repro_v1/bin/python bash run_all.sh

# 접속이 끊겨도 계속 돌게 하려면
nohup env PY=~/venv_repro_v1/bin/python bash run_all.sh > run_all.console 2>&1 &
```

선택 환경변수:
- `WORKERS`: Fourier–Gram 병렬 프로세스 수다. 기본값은 min(8, CPU 수)이고, 결과 값에는 영향을 주지 않는다.
- `OUT_ROOT`: 결과 위치다. 기본값은 `./runs`다.

`run_all.sh`가 하는 일:
1. `OMP/OPENBLAS/MKL/NUMEXPR/VECLIB/RAYON_NUM_THREADS=1`, `PYTHONHASHSEED=0`, `PYTHONDONTWRITEBYTECODE=1`, `PYTHONINTMAXSTRDIGITS=0`으로 고정한다.
2. python, numpy, sympy, qiskit 버전(mpmath, scipy, rustworkx도 함께)을 출력한다. 3.8.10, 1.24.4, 1.13.3, 1.2.4와 다르면 **경고만 하고 계속 진행한다.**
3. `BUNDLE_MANIFEST.json`으로 번들 파일 무결성을 확인한다.
4. 단위 테스트를 실행한다. Fourier–Gram 11개, Kyriienko 14개다.
5. (A) `run_experiment.py --workers $WORKERS`를 실행한다.
6. (B) `kyriienko_scaling.py --sizes 2 4 6 8`을 실행한다. 이 스크립트는 끝에서 저장된 Jacobian으로 행렬식을 다시 계산하는 audit replay도 수행한다.
7. `compare_results.py`로 참조 결과와 비교하고 `comparison_verdict.json`을 쓴다.
8. `runs/server_return_<UTC>.tar.gz`로 압축한다.

종료 코드는 PASS 또는 PASS_WITH_WARNINGS이면 0, FAIL이면 1이다. 단계 하나가 실패해도 나머지 단계와 비교는 끝까지 실행하므로, FAIL인 경우에도 압축 파일은 만들어진다.

## 3. 예상 실행 시간과 메모리

로컬 측정값이다(macOS arm64, 18코어, `WORKERS=8`). 세 환경의 결과는 모두 `local_validation/`에 있다(Py 3.9.6 + SymPy 1.13.3 실행은 255 s였다).

| 단계 | 기존 venv (Py 3.9.6, SymPy 1.14.0) | 서버 유사 venv (Py 3.8.20, SymPy 1.13.3) | 최대 RSS (가장 큰 단일 프로세스) |
| --- | --- | --- | --- |
| 단위 테스트 (두 종류 합계) | 2.9 s | 2.6 s | 약 120 MB |
| (A) Fourier–Gram 30개 | 223 s | 267 s | 1.34 GB |
| (B) Kyriienko n = 2–8 | 24.6 s | 24.7 s | 213 MB |
| 버전 점검 + 무결성 + 비교 + 압축 | 수 초 | 수 초 | 약 100 MB |
| **전체** | **251 s (약 4.2분)** | **295 s (약 4.9분)** | |

- (A)의 직렬 CPU 시간은 약 1,230 s이고, 가장 오래 걸리는 단일 설정은 약 160 s다. 따라서 wall time은 대략 max(160 s, 1,230 s / WORKERS)다.
  예를 들어 8개면 약 4–5분, 4개면 약 6분, 1개면 약 20분이다. x86_64 서버는 클럭에 따라 1–2배 더 걸릴 수 있다.
- 메모리: (A)의 단일 프로세스는 최대 약 1.34 GB를 쓴다. 60만 항 기호 전개 때문이다.
  WORKERS = 8일 때 주 프로세스와 worker 전체 RSS 합계의 최댓값은 **약 6.3 GB**였다(`local_validation/fourier_memory_probe_8workers.txt`).
  여유 메모리가 8 GB보다 적으면 `WORKERS=4`(추정 4 GB 이하)를 쓴다. (B)는 단일 프로세스이고 약 0.2 GB를 쓴다.
- 디스크: 압축 해제한 번들은 약 6 MB이고, 실행 결과는 회당 약 2.4 MB다(압축 1.6 MB).

## 4. 돌려보낼 파일

**`runs/server_return_<UTC>.tar.gz`** 하나(약 1.6 MB)를 보내면 된다. 압축이 실패했으면 `runs/<UTC>/` 폴더 전체를 보낸다. 안에는 다음이 들어 있다.

- `comparison_verdict.json`: 최종 판정, 위반 항목, 설정별 표.
- `environment_check.json`: 버전, 플랫폼, CPU 수, 스레드 환경변수.
- `bundle_integrity.json`, `stage_status.json`: 단계별 종료 코드, wall time, 최대 RSS.
- `total_wall_seconds.txt`.
- `logs/`: `run_all.log`와 단계별 로그.
- `fourier_gram/`: 새 실행 결과 전체(설정별 JSON 30개, summary, provenance, manifest).
- `kyriienko_scaling/`: 새 실행 결과 전체(설정별 JSON 12개, Jacobian 12개, `ir/` 12개, validation_gate, audit_replay 등).

## 5. PASS의 의미

`compare_results.py`는 모든 점검을 세 등급으로 나눈다. 로컬 실행에서는 점검 약 3,700개 중 strict가 3,376개였다.

**strict**: 하나라도 다르면 FAIL이다.
- (A)
  - 설정마다 `r_lower`, `r_upper`, status, 원고와의 consistency를 비교한다.
  - θ* 지정(Pythagorean 삼중쌍)을 비교한다.
  - 두 소수 각각에서 모듈러 랭크, 소행렬 행·열, 행렬식 mod p, active witness를 비교한다.
  - 정확 유리수 소행렬식 7개를 문자열과 `Fraction` 값으로 각각 비교한다.
  - U1 삭제 로그, 가상 field(기호식은 SymPy로 `expand(a−b) == 0` 확인), 기호별 판정 목록과 개수, 예산 제한 32개 entry의 판정을 비교한다.
  - 코드, 동결 입력, 원고 표의 해시를 비교한다.
- (B)
  - 생성된 IR 12개의 SHA256을 비교한다. n = 2, 4의 6개는 동결 IR과의 바이트 일치도 확인한다.
  - 설정마다 인증 랭크, 구간, status를 비교한다.
  - 정수·유리수 소행렬식과 행 스케일 곱을 문자열과 정확한 값으로 각각 비교한다. 세 소수 나머지, 소행렬 행·열도 비교한다.
  - 정수 Jacobian 내용의 SHA256을 비교한다.
  - 삭제 좌표, 가상 field, θ*에서의 정확 null vector, 기호 개수(삭제/가상 field/소행렬 열)를 비교한다.
  - validation gate, Euler 항등식, audit replay, 코드와 core 해시를 비교한다.
- 공통: 단계 종료 코드 4개가 모두 0이어야 하고, 번들 무결성 점검을 통과해야 한다. 새 결과 폴더의 자체 manifest 해시도 맞아야 한다.

**warn**: PASS_WITH_WARNINGS가 된다.
- 인증이 아닌 수치 진단이다. float SVD 랭크, float/dense/Qiskit statevector 교차검증(허용 1e−10), interval 엔진 교차검증이 여기에 속한다.
- 버전 불일치도 warn이다. 이 번들의 참조 결과는 Python 3.9.6/SymPy 1.14.0으로 만들었으므로, 서버(3.8.10/1.13.3)에서는 항상 python·sympy 버전 경고가 나온다. **이것은 예상된 경고다.**

**info**: 판정에 영향을 주지 않는다.
- (A)의 기호 two-copy Gram 전개는 60 s 또는 60만 항 예산 안에서만 시도한다. 따라서 기계 속도에 따라 완료 여부와 그에 딸린 certificate 목록, `attained_by`, 완료 설정 수가 달라질 수 있다.
- 이 값들은 인증 결과를 바꾸지 않는다. INACTIVE 57개는 모두 예산과 무관한 two-copy 불변 삭제 인증으로 판정된다. U2가 필요한 유일한 상한(Hubregtsen n4 L1 Z)은 0.1 s 안에 끝난다.
- 같은 식의 SymPy 문자열 표현이 다른 경우도 info다. 수학적 동일성은 strict로 따로 확인한다.

**PASS가 뜻하는 것**: 서버의 다른 플랫폼과 버전에서 같은 코드를 실행했을 때, 인증에 쓰인 정확한 양(랭크, 행렬식, 판정, IR)이 참조 실행과 비트 단위로 같았다는 뜻이다. 이는 구현과 환경 의존성 점검이며, 정리의 독립적인 증명은 아니다(7절).

## 6. SymPy 1.13.3과 Python 3.8 점검 결과 (로컬)

PyPI에 접근할 수 있어서, 번들 밖 scratch 영역에 임시 venv를 만들고 `run_all.sh` 전체를 실제로 실행했다.
Python 3.8은 `uv`가 제공하는 python-build-standalone 3.8.20을 썼다. 패키지는 `requirements_server.txt`의 pin대로 pip로 설치했다.
각 실행의 판정 JSON, 단계 상태, 환경, 전체 로그는 `local_validation/<환경>/`에 있다.

| 환경 | 결과 |
| --- | --- |
| 기존 `.venv_compiler_v2`: Py 3.9.6, NumPy 1.24.4, SymPy 1.14.0, Qiskit 1.2.4 | **PASS_WITH_WARNINGS**, 251 s. strict 3,376개 모두 일치. 경고는 python·sympy 버전 2개뿐이고 info 차이는 0개다 |
| 서버 유사: **Py 3.8.20**, NumPy 1.24.4, **SymPy 1.13.3**, Qiskit 1.2.4, SciPy 1.10.1, mpmath 1.3.0, rustworkx 0.15.1 | **PASS_WITH_WARNINGS**, 295 s. strict 3,376개 모두 일치. 경고는 python 3.8.20 ≠ 3.8.10 하나뿐이다(gcd shim 사용) |
| SymPy만 변경: Py 3.9.6, SymPy 1.13.3 (나머지 동일) | **PASS_WITH_WARNINGS**, 255 s. strict 3,376개 모두 일치. 경고는 python 버전 하나뿐이다 |
| 음성 대조: 새 결과 사본에서 행렬식 한 자리, 기호 판정 하나, IR 1바이트를 바꾸고, 같은 식을 다른 순서의 SymPy 문자열로 바꿈 | **FAIL**. 변조 3개가 모두 strict로 잡혔다. 순서만 바꾼 식은 info로만 기록됐고 기호 동일성 점검은 통과했다(`local_validation/negative_control_tampered_copy/`) |

세 실행 모두 30개 설정의 Gram 랭크, 정확 소행렬식 7개, 모듈러 소행렬식 60개, 기호별 판정(ACTIVE 195, INACTIVE 57, UNDECIDED 0),
Kyriienko 인증 랭크 12개, 정수·유리수 소행렬식 12개, IR SHA256 12개, Jacobian SHA256 12개, validation gate, audit replay가 참조와 같았다.
기호 two-copy 전개의 예산 내 완료 여부도 세 실행 모두 참조와 같았다(info 차이 0개).

- **SymPy 1.13.3**: 두 계산 모두 정상 실행됐다. 인증 양은 전부 참조와 같았다.
  사용한 API(`Matrix`, `symbols`, `sympify`, `expand`, `simplify`, `reduced`(Gröbner), `nsimplify`, `lambdify`, `DomainMatrix`/`ZZ`, `expand_trig` 등)에서 결과에 영향을 주는 차이는 발견되지 않았다.
  Qiskit `ParameterExpression`에서 IR 문자열을 만드는 경로도 1.13.3과 1.14.0에서 같은 바이트를 냈다(IR SHA256 12/12 일치).
- **Python 3.8 호환성 문제(발견 후 우회)**: `kyriienko_scaling.py`는 `math.gcd(A, B, C)`와 `math.gcd(*row)`를 쓴다. 가변 인자 gcd는 Python 3.9부터 지원되므로, 3.8에서는 원본 그대로 실행하면 `TypeError: gcd expected 2 arguments, got 3`로 테스트 6개가 실패한다.
  번들은 원본 파일을 고치지 않는다. 대신 `tools/repro_helper.py pyexec`가 Python < 3.9일 때만 `math.gcd`를 3.9와 의미가 같은 가변 인자 버전으로 바꾼 뒤 스크립트를 실행한다(gcd() = 0, gcd(a) = |a|, 그 외에는 2인자 gcd를 차례로 적용).
  Fourier–Gram 코드는 2인자 gcd만 쓰므로 그대로 실행한다.
- **정수↔문자열 자릿수 제한**: CPython 3.8.14 이후의 패치 릴리스(3.8.20 포함)는 4300자리를 넘는 int↔str 변환을 막는다. 행렬식은 최대 약 16,000자리다.
  `run_all.sh`는 `PYTHONINTMAXSTRDIGITS=0`을 설정하고, `compare_results.py`도 같은 제한을 해제한다. 서버의 3.8.10에는 이 제한이 없으므로 영향이 없다.
- 로컬 Python은 3.8.20이고 macOS arm64다. 서버는 3.8.10, Linux x86_64이므로 완전히 같은 환경은 아니다.

## 7. 한계

- **형식 검증이 아니다.** 산술은 Python 정수, `fractions`, NumPy object 배열, SymPy(Gröbner 약분, DomainMatrix)로 한다. 증명 보조기로 검증한 것이 아니다.
  이 번들은 같은 코드를 다른 기계와 버전에서 다시 실행해 출력이 같은지 보는 재현 점검이다. 독립 재구현이 아니다.
- **transpiler 의존성.** (B)의 IR은 Qiskit 1.2.4 `transpile`(optimization_level 3, seed_transpiler 71, approximation_degree 1, census basis)의 결과다. Qiskit 버전, rustworkx, 플랫폼 부동소수가 달라지면 게이트 열이 달라질 수 있다.
  예전 서버 census(2026-10-01)에서도 highdim 회로의 상수 각도가 서버와 로컬에서 달랐던 적이 있다(Kyriienko는 동일했다). 이런 차이는 IR SHA256 점검에서 FAIL로 드러난다.
  그 경우 `comparison_verdict.json`의 `per_setting` 표를 보면 랭크 자체가 같은지 따로 확인할 수 있다. n = 6, 8 IR은 원천 공식과 대조하는 구조 점검도 함께 받는다.
  (A)는 Qiskit을 쓰지 않고 동결 IR만 읽는다.
- **예산 의존 항목.** 5절의 info 항목은 기계 속도에 따라 달라질 수 있다. 다만 인증 판정은 바뀌지 않도록 설계했다.
- **수치 진단.** float 값(SVD 특이값, 교차검증 오차)은 BLAS와 플랫폼에 따라 1e−15 수준에서 달라질 수 있다. 그래서 값 자체는 비교하지 않고 통과 여부와 랭크만 비교한다.
- **수학적 범위.** 하한은 한 점 θ*에서의 정확 소행렬식이고, 랭크의 하반연속성으로 generic 랭크의 하한이 된다. (B)는 짝수 n = 2, 4, 6, 8만 다루며 모든 n에 대한 정리가 아니다.
  (B)의 XYZ와 weight≤2 상한은 원고 정리(Eq. kyrifamilyupper)에 기대고, 이 번들은 그 가정과 gauge 항등식을 IR에서 재검증한다.
- 참조 결과는 macOS arm64, Python 3.9.6, SymPy 1.14.0에서 만들었다.

## 8. 문제 해결

- `Qiskit ... MISSING` 경고가 나오면 (B)가 실패한다. `PY`가 올바른 venv를 가리키는지 확인한다.
- FAIL이면 `comparison_verdict.json`의 `strict_failures` 목록을 본다. 항목마다 section, check, 참조값, 새 값(긴 값은 길이와 해시)이 있다. 결과는 그대로 압축해서 보내면 된다.
- 같은 번들에서 여러 번 실행해도 된다. 실행마다 `runs/<UTC>/` 폴더가 새로 생긴다. 두 실행을 동시에 돌리면 (A)의 결과 폴더 감지가 섞일 수 있으므로 동시에 돌리지 않는다.
- 비교만 다시 하려면 `python3 compare_results.py --run-dir runs/<UTC>`를 실행한다.
