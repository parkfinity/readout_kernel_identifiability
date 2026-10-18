# Kyriienko형 양자 커널의 generic 미분 랭크: n = 6, 8 정확 인증

실행: 2026-10-03 로컬, `results/20261003T121202Z/`. 단일 프로세스이고, 테스트 14개를 포함한 전체 wall time은 42 s(실험만 32 s), peak RSS는 225 MB다.
환경: `../.venv_compiler_v2` (Python 3.9.6, **Qiskit 1.2.4**, NumPy 1.24.4, SymPy 1.14.0, SciPy 1.10.1, macOS arm64).
버전 정보는 `environment.json`에 있다. 동결 census의 `requirements_py38.txt`는 SymPy 1.13.3을 지정하지만, 여기서는 동결 확장 실행의
로컬 환경과 같은 SymPy 1.14.0을 썼다.

## 결론

| n | K = 3n | 3n−2 | local XYZ | weight ≤ 2 | local Z (보조) |
| --- | --- | --- | --- | --- | --- |
| 2 | 6 | 4 | 4 | 0 (k = n, 완전 Pauli) | 4 |
| 4 | 12 | 10 | 10 | 10 | 8 |
| **6** | 18 | 16 | **16** | **16** | **12** |
| **8** | 24 | 22 | **22** | **22** | **16** |

- 모든 칸은 정확 하한과 상한이 일치해 인증됐다(`CERTIFIED_EQUAL`). 수치 랭크를 대신 쓴 칸은 없다.
- **n = 6, 8에서 local XYZ와 weight ≤ 2 모두 r = 3n − 2가 성립한다.** 이 결과는 n = 2, 4, 6, 8이라는 네 개의 짝수 크기에서
  확인한 것이고, 모든 n에 대한 증명은 아니다.
- local Z는 n = 2, 4, 6, 8에서 r = 2n이다. 상한은 원고가 아니라 이 실행의 기계 인증에서 나왔다(아래 4절).

## 1. 검증 게이트 (통과)

- **IR 재현.** 복사한 동결 코어로 n = 2, 4의 IR 6개를 만들었다. 이 코어는 `core_copy/`에 있으며 32개 파일이 원본과 SHA256 기준으로 같다.
  만든 IR은 `certified_dimension_extension_v1/frozen_inputs`와 **바이트 단위로 같고**, SHA256도 `scope.json`과 일치한다.
  구성 경로는 vendored sQUlearn 0f6cb098, `KyriienkoEncodingCircuit(num_qubits=n)` 기본값,
  `generate_initial_parameters(num_features=2, seed=0)`, transpile(opt=3, seed 71, approximation_degree=1, 1 process,
  census basis), `adapter.adapt`, `program(fuse_adjacent=True)`다.
- **랭크 재현.** 1차 값은 n2 XYZ 4, n4 XYZ 10, n4 weight≤2 10이다. 보조 값은 n2 Z 4, n4 Z 8, n2 weight≤2 0이다.
  여섯 값 모두 같은 엔진이 정확하게 재현했다(`validation_gate.json`).
  단위 테스트도 동결 IR 파일을 직접 읽어 같은 값을 다시 확인한다.

## 2. n = 6, 8 회로 (원천 공식과 대조)

- IR은 Eq. (kyriform)의 형태다. 각 큐비트에 R_y(κ_q·arccos x_{q mod 2})를 걸고, 독립 파라미터 θ_{3q},
  θ_{3q+1}, θ_{3q+2}로 RZ·RX·RZ를 적용한다. 그 뒤 전방 최근접 CX를 짝수 시작 쌍, 홀수 시작 쌍 순서로 건다.
- 원천 공식에서 Qiskit과 IR을 거치지 않고 직접 만든 wire별 게이트 열이 IR과 정확히 같다.
  공식은 m번째 인코딩 게이트를 qubit m mod n에 걸고, 각도를 2(⌊m/2⌋+1)·arccos x_{m mod 2}로 둔다.
  여기서 **κ_q = 4⌊q/2⌋ + n + 4**이고 feature는 q mod 2다.
  n = 6이면 κ = (10,10,14,14,18,18), n = 8이면 κ = (12,12,16,16,20,20,24,24)다.
- 정확 대수로 다음을 확인했다. E는 Z_0, X_{n−1}과 교환한다. 모든 설정이 fixed-frequency다. 단위 가중치이고, 각 θ는 한 번만 나온다.

## 3. 하한 (정확 소행렬식)

- 점: θ_k = 4·arctan(u_k), u_k = (k+2)/(k+5)이다. 이는 2·arctan(t_k), t_k = 2u_k/(1−u_k²)와 같다.
  **반각 θ_k/2 = 2·arctan u_k**이므로 e^{iθ_k/2} = (1+iu_k)²/(1+u_k²)는 가우스 유리수다.
  입력은 x_b = cos φ_b, φ_b = 2·arctan s_b이고, s_b > 0이므로 φ_b ∈ (0, π)와 arccos x_b = φ_b가 정확히 성립한다.
  데이터 반각은 (κ/2)·φ_b이고 κ가 짝수이므로 정수배다. 코드는 이를 검사한다.
  입력점은 10개다. `random.Random(20261003)`로 작은 유리수 s 목록에서 뽑았고, 모든 쌍 45개를 행으로 쓴다.
- 엔진은 공통 정수 분모를 가진 가우스 정수 상태벡터와 그 θ-접벡터다. 원소 J[pair, k] = ∂_k d²(x_i, x_j)를
  정확한 유리수로 계산한다. 각 행에 양의 유리수를 곱해 정수 행으로 만든다. 행렬식의 0 여부는 이 변환에서 변하지 않는다.
- 소행렬식은 mod p = 2⁶¹−1 소거로 골랐다. 행렬식은 Bareiss로 정확히 계산했고, SymPy DomainMatrix(ZZ)가 같은 값을 냈다.
  세 소수 2⁶¹−1, 2⁶²−57, 2⁶³−25에 대한 나머지도 모두 0이 아니다.
  `audit_replay.json`은 저장된 Jacobian 파일만으로 행렬식을 다시 계산했고, 결과가 일치했다.

| 설정 | 차수 | 제외 열 | 정수 소행렬식 자릿수 | 실제 유리수 소행렬식 |
| --- | --- | --- | --- | --- |
| n6 XYZ | 16 | θ2, θ17 | 5773 | ≈ −0.99 |
| n6 weight≤2 | 16 | θ2, θ17 | 7093 | ≈ 1.6×10⁴ |
| n6 Z | 12 | θ2,5,8,11,14,17 | 2823 | ≈ 1.7×10⁻⁶ |
| n8 XYZ | 22 | θ2, θ23 | 10464 | ≈ −0.094 |
| n8 weight≤2 | 22 | θ2, θ23 | 15632 | ≈ −1.5×10⁷ |
| n8 Z | 16 | θ2,5,…,23 | 5346 | ≈ −7.4×10⁻³ |

  정확한 정수·유리수 행렬식, 행과 열, 점, 쌍은 설정별 JSON에 있다. 전체 정수 Jacobian은 `*_jacobian.json.gz`에 있다.
  랭크는 하반연속이므로 한 점의 0이 아닌 소행렬식은 generic r ≥ 차수를 증명한다.

## 4. 상한

- **XYZ와 weight ≤ 2 (k = 2 < n).** 원고 `family_extension.tex` Eq. (kyrifamilyupper)가 r ≤ 3n−2를 준다.
  같은 논증을 생성된 IR에서 기계로 다시 확인했다.
  (a) θ_2는 정확 삭제된다. 수송된 생성원은 Z_0이고, [Ω, P⊗I + I⊗P] = 0을 정확한 Pauli 대수로 확인했다.
  (b) 마지막 큐비트 Euler 블록 뒤에 가상 X를 삽입하면 수송된 생성원은 X_{n−1}이고, 같은 two-copy 불변성이 성립한다.
  Cramer field v = (sin c, cos c sin b, −sin c cos b)에 대해 Σ v_k ∂_k U = −(i/2) sin b·X·U이며,
  이 항등식은 SymPy로 기호 검증했다. 대조로 성분 하나를 2배로 바꾼 field는 거부된다.
  두 field는 θ*에서 정확 랭크 2이므로 r ≤ K − 2 = 3n − 2다.
- **local Z.** 원고에는 해당 상한이 없다. 대신 각 큐비트의 마지막 RZ 각도 θ_{3q+2} n개를 정확 삭제한다.
  이들의 출력 생성원은 Z 문자열이고 모든 Z readout과 교환하므로 r ≤ 2n이다. 0이 아닌 2n차 소행렬식과 합치면 r = 2n이 인증된다.
- **n = 2 weight ≤ 2**는 완전 Pauli이므로 6개 각도가 모두 삭제되고 r = 0이다. 원고의 k = n 문장과 일치한다.
- **θ*에서의 일관성 점검.** 모든 정확 null 벡터가 45개 정확 행 전부를 소거한다. 따라서 θ*에서 정확 랭크는 인증값과 같다.
  이는 한 점에서의 점검이며 별도의 증명은 아니다.

## 5. 수치 점검 (인증 아님)

- SVD 랭크: 무작위 θ 4개 각각에서 max(3K, 60)개의 무작위 쌍을 썼고, 상대 tol은 1e−9다. 12개 설정 모두 인증값과 같다.
  σ_r/σ_ref는 7×10⁻⁶ ~ 0.36이고, σ_{r+1}/σ_ref는 2×10⁻¹⁸ ~ 4.4×10⁻¹⁶이다.
- gauge field: 무작위 점에서 상대 잔차는 최대 2.4×10⁻¹⁶이다. 교란한 대조 field의 잔차는 최소 2×10⁻²이다.
  예외는 n2 weight≤2로, J 자체가 0이라서 대조도 0이다.
- 엔진 교차검증: θ*에서 정확 특징과 float IR 엔진의 차이는 9.2×10⁻¹⁵ 이하다.
  IR을 쓰지 않는 독립 dense statevector(텐서 축약)와 Qiskit 원천 Statevector와의 차이는 각각 2.2×10⁻¹⁵, 1.9×10⁻¹⁵ 이하다.
  정확 기울기와 float 접벡터의 차이는 2.2×10⁻¹⁵ 이하다.
  단위 테스트는 무작위 점에서 d²를 tol 1e−10으로 비교하고, 유한차분 기울기를 1e−7로 비교한다.

## 6. 실행 시간·메모리

| n | IR 생성 | 정확 상태·특징 (10점, weight≤2 합집합 공유) | Bareiss / SymPy det (XYZ, w≤2) | 설정 합계 |
| --- | --- | --- | --- | --- |
| 6 | 0.02 s | 0.7 s | 0.12 / 0.16 s, 0.18 / 0.27 s | 2.4 s |
| 8 | 0.02 s | 12.5 s | 0.96 / 1.86 s, 2.16 / 3.59 s | 28 s |

실험 전체 wall time은 32.4 s이고, peak RSS는 225 MB다. 설정별 시간은 각 JSON의 `timings_seconds`에 있다.
정확 상태 계산 시간은 각 n에서 처음 처리한 readout에 몰아서 기록했다.

## 한계

- **모든 n에 대한 정리가 아니다.** 확인한 크기는 짝수 n = 2, 4, 6, 8뿐이다. 홀수 n에서는 qubit q가 두 feature를 모두 받으므로
  (q와 q+n의 parity가 다르다) 구조가 다르며, 시험하지 않았다. 깊이, 인코딩 스타일, ABA arrangement, feature 수도 기본값 하나만 다뤘다.
- 하한은 θ* 한 점에서 얻었다. 다른 특수점에서는 랭크가 더 작을 수 있으며, 이는 generic 정의와 맞다.
- 산술 스택은 증명 보조기로 검증된 것이 아니다. 대상은 Python 정수, NumPy object 배열, SymPy simplify/DomainMatrix다.
  Bareiss와 SymPy 행렬식의 일치, 세 소수 나머지, 독립 float·dense·Qiskit 엔진의 일치는 구현 점검이지 형식 증명이 아니다.
- XYZ와 weight≤2 상한의 근거는 원고 정리다. 여기서는 그 가정과 두 gauge 항등식을 IR에서 재검증했다.
  local Z 상한은 이 실행의 삭제 인증이고 원고에는 없다.
- n = 6, 8 IR은 Qiskit 1.2.4 transpile 결과에 의존한다. 다만 원천 공식 대조와 Qiskit 없는 dense statevector가 이를 독립적으로 확인한다.
- r은 커널 미분 랭크다. 게이트 수, shot 수, 학습 가능성, 전역 최소 chart에 대한 주장이 아니다.
- `results/superseded/20261003T120950Z`는 최종 전 실행이다. 이후 assertion과 audit replay만 추가했고, 시간을 제외한 표 값은 최종 실행과 같다.

## 파일

- `kyriienko_scaling.py`: source→IR, 구조 점검, 정확 엔진, 정확 선형대수, 상한 인증, float/dense/Qiskit 교차검증, 실행기,
  audit(`--verify`).
- `test_kyriienko_scaling.py`: 단위 테스트 14개. IR 재현, 랭크 재현, 엔진 일치(1e−10), 유한차분, Euler 항등식,
  two-copy 대조, 선형대수, 음성 대조를 포함한다.
- `run.sh`: `test | gate | full | verify <dir>`. Qiskit 1.2.4가 아니면 중단한다.
- `core_copy/`: `cross_family_resource_v1/core`의 바이트 동일 사본이다(해시는 `provenance.json`).
- `results/20261003T121202Z/`: 설정별 JSON 12개와 Jacobian 12개, `ir/`, `summary.json`, `validation_gate.json`,
  `euler_identity.json`, `environment.json`, `provenance.json`, `manifest.json`, `audit_replay.json`, `log.txt`.
