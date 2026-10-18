# Fourier–Gram 정확 랭크 인증: 30개 확장 설정 결과

실행: 2026-10-02 로컬, `results/20261002T044925Z/` (wall 195 s, 8 프로세스).
환경: Python 3.9.6, NumPy 1.26.4, SymPy 1.14.0, mpmath 1.3.0, macOS arm64 (`environment.json`).
입력: `certified_dimension_extension_v1/frozen_inputs/`의 30개 IR을 읽기만 했다. 30개 파일 SHA256과
`ie.info_hash`가 모두 `scope.json`과 일치한다. 재사용한 `exact_engine.py`, `interval_engine.py`는
수정 없이 `copied_from_extension_v1/`로 복사했고 원본과 해시가 같다 (`provenance.json`).

## 요약

- **30개 설정이 모두 fixed-frequency이다.** 학습 기호와 데이터를 곱하는 항 `[c,t,x]` (t≥0, x≥0)이 0개이고,
  상수 각도 항도 없다. YZ-CX는 `R_P(θ+x)` 형태의 가산 각도이고(곱이 아님), Hubregtsen은 `R_z(x)`와
  ±1/2 계수의 CRZ 분해를 쓴다. Kyriienko는 데이터가 `R_y(κ·arccos x)`로만 들어가고(동결 IR 기준 κ:
  n=2에서 2+4=6, n=4에서 큐비트 0,1은 2+6=8, 큐비트 2,3은 4+8=12), 변분층 RZ·RX·RZ는 순수 기호 각도다.
- **열려 있던 Kyriienko 구간 3개가 모두 닫혔다.**

| 설정 | 원고 | Gram 결과 | 하한 인증 | 상한 인증 |
| --- | --- | --- | --- | --- |
| Kyriienko n=2 L=1 local XYZ | [4,5] | **4** | θ*에서 정확 유리수 4×4 소행렬식 ≈ −7.17×10⁻⁵ (열 θ0,θ1,θ3,θ4), 두 소수 mod p에서도 0 아님 | null field 2개: e₂, 그리고 큐비트 1 Euler 블록 뒤 가상 X 삽입 (출력 생성원 IX) |
| Kyriienko n=4 L=1 local XYZ | [10,11] | **10** | 정확 유리수 10×10 소행렬식 ≈ −1.19×10⁻²⁴ (열 θ2, θ11 제외) | e₂, 큐비트 3 블록 뒤 가상 X 삽입 (출력 IIIX) |
| Kyriienko n=4 L=1 weight≤2 | [10,11] | **10** | 정확 유리수 10×10 소행렬식 ≈ 6.41×10⁻¹⁸ | e₂, 큐비트 3 블록 뒤 가상 X 삽입 (출력 IIIX) |

  소행렬식은 근삿값이 아니라 1000자리 안팎의 정확한 유리수로 per-setting JSON에 저장했다. 크기가 작은
  것은 여러 작은 계수의 곱이기 때문이며 인증과 무관하다.
- **원고의 27개 확정 랭크와 30개 모두 일치한다** (불일치 0). 수치 SVD 랭크도 30개 모두 인증값과 같다(인증 아님).
- **예산 제한 32개 entry는 모두 INACTIVE로 정확히 판정됐다.** 32개 전부 two-copy 불변 삭제 인증이 있고,
  그중 9개는 θ를 기호로 둔 Gram 전개만으로도 `∂_k G ≡ 0`이 나왔다. 활성 증거와 충돌한 경우는 없다.
- 기호별 판정은 활성 195, 비활성 57, 미판정 0이다. 기존 structural closure의 252개 판정과 하나도 다르지 않다.

## 방법 (운영 핵심)

1. **Fourier 기저.** 각 데이터 기저 φ_b(x) ∈ {x_j, arccos x_j}에 대해 ψ_b = φ_b/M_b로 두면, 모든 회전은
   진폭에 u_b^{±s}(u_b = e^{iψ_b}, s는 정수)를 곱한다. 따라서 특징은 정수 주파수 |w_b| ≤ 2S_b의
   삼각다항식이고, f_O = c_O(θ) + Σ_a C_{Oa}(θ) h_a(x), h = {cos(w·ψ), sin(w·ψ) : w ∈ Ω₊}이다.
   상수 모드는 제거한다. 30개 모두 두 기저가 서로 다른 입력 좌표 위에 있으므로, 열린 영역에서
   1과 h_a는 선형독립이다(해석적 삼각다항식이 열린 집합에서 0이면 항등적으로 0).
2. **주입성 보조정리.** {1, h}가 독립이면 d² = Δhᵀ G Δh (G = CᵀWC)는 G를 결정한다. 따라서 모든 θ에서
   rank J_{d²}(θ) = rank J_G(θ)이고, 데이터 변수는 계산에서 완전히 빠진다.
3. **정확 C(θ*) 계산.** 각 기호에 Pythagorean 삼중쌍을 배정해 e^{iθ_k/L_k} = (a+ib)/c인 정확한 θ*를 고른다.
   예를 들어 θ₀ = 2·atan2(−21,−220)이다. 그다음 상태를 Q(i)[u^{±1}] 위의 Laurent 다항식으로 전파한다.
   H는 √2H 정수행렬로 처리하고, 이차 특징에서 2^{−#H}를 복원한다. 모든 설정은 두 소수 p로 나눈
   나머지에서 계산하고, Kyriienko 6개와 Hubregtsen n4 L1 Z는 정확 유리수로도 계산한다.
   두 정확 엔진은 mod p에서 계수가 완전히 같다(불일치 0). 독립 float statevector와 aliasing 없는 DFT
   격자 결과와의 차이도 8.9×10⁻¹⁶ 이하다.
4. **하한.** θ*에서 정확한 유리수 J_G의 r×r 소행렬식이 0이 아님을 보인다. 정수 소행렬식이 mod p에서
   0이 아니면 정수로도 0이 아니므로 엄밀하다. 일곱 설정은 정확 유리수 행렬식도 함께 제시한다.
   랭크는 하반연속이므로 generic rank ≥ r이다.
5. **상한 U1: two-copy 불변 null field.** Ω = Σ_O w_O O⊗O라 두면 K(x,x′) = Tr[(ρ(x)⊗ρ(x′))Ω]이고,
   d²와 Gram 형식은 모두 이 양에서 정해진다.
   (a) 출력까지 수송한 생성원 P가 [Ω, P⊗I + I⊗P] = 0(공통 θ)이거나 [Ω, P⊗I] = [Ω, I⊗P] = 0(데이터
   각도)을 만족하면 그 회전을 반복 삭제한다. Pauli 대수와 위상은 정확하게 계산한다.
   (b) 세 회전 Euler 블록 뒤에 단일 큐비트 Q를 가상으로 삽입한다. 수송한 생성원이 Ω의 공통 대칭이면,
   Cramer field v = adj(M)e_Q가 모든 θ에서 J(θ)v(θ) = 0을 만족한다. 이 2×2 행렬 항등식은
   {C²+S²−1, i²+1} Gröbner 기저에 대한 정확한 다항식 약분으로 확인했다. ZXZ 블록과 Q=X의 경우
   v = (sin θ_c, cos θ_c sin θ_b, −sin θ_c cos θ_b)이고 행렬식은 sin θ_b다.
   (c) 이 field 집합의 θ*에서의 정확 랭크 m으로 r ≤ K − m을 얻는다.
6. **독립 기호 확인 (수치 아님).** 모든 삭제 단계는 해당 게이트 각도를 새 변수로 바꾼 뒤 light-cone 꼬리의
   two-copy 연산자를 기호 전개해, 그 변수에 의존하지 않음을 확인했다. 81개 단계 모두 확인됐다.
   모든 가상 field는 꼬리 연산자에서 Σ_k v_k ∂_k Ω_suffix ≡ 0이 성립하고, 성분 하나를 2배로 바꾼
   field는 거부되는지 함께 확인했다. 3개 모두 통과했다. n=2 XYZ에서는 θ 전체를 기호로 둔 Gram 전개에서도
   Σ v_k ∂_k Ĝ ≡ 0을 확인했다. J(θ*)v(θ*) = 0 mod p는 일관성 점검일 뿐 증명으로 쓰지 않았다.
7. **상한 U2: 기호 Gram.** two-copy Heisenberg 전개로 K(x,x′;θ)를 θ에 대한 기호식으로 얻는다
   (예산: 60 s, 60만 항). 여기서 기호별 `∂_k G ≡ 0`과 r ≤ dim span{G_ab − 상수}를 얻는다.
   Hubregtsen n4 L1 Z의 상한 2(β₀=sin²θ₀+sin²θ₂, β₁=sin²θ₁+sin²θ₃)는 이 방법으로 닫힌다.

## 원고 표와의 일관성 (r = Gram 하한 = 상한)

| 계열 | n | L | K | XYZ (원고 / Gram) | Z | weight≤2 |
| --- | --- | --- | --- | --- | --- | --- |
| YZ-CX | 2 | 1 | 4 | 3 / 3 | 2 / 2 | 2 / 2 |
| YZ-CX | 2 | 2 | 8 | 7 / 7 | 6 / 6 | 6 / 6 |
| YZ-CX | 4 | 1 | 8 | 6 / 6 | 4 / 4 | 6 / 6 |
| YZ-CX | 4 | 2 | 16 | 14 / 14 | 12 / 12 | 14 / 14 |
| Hubregtsen | 2 | 1 | 2 | 0 / 0 | 2 / 2 | 0 / 0 |
| Hubregtsen | 2 | 2 | 4 | 2 / 2 | 4 / 4 | 2 / 2 |
| Hubregtsen | 4 | 1 | 8 | 8 / 8 | 2† / 2 (U2) | 8 / 8 |
| Hubregtsen | 4 | 2 | 16 | 16 / 16 | 12 / 12 | 16 / 16 |
| Kyriienko | 2 | 1 | 6 | [4,5] / **4** | 4 / 4 | 0 / 0 |
| Kyriienko | 4 | 1 | 12 | [10,11] / **10** | 8 / 8 | [10,11] / **10** |

U2라고 표시한 칸을 제외하면 상한은 U1(삭제 + 가상 field) 또는 자명한 K에서 온다. Kyriienko 구간 셋은
가상 field가 있어야 닫힌다. 그 밖의 26개는 원고와 같은 메커니즘인 상수 null vector로 닫힌다.

교차검증(모두 통과)
- 설정마다 무작위 32점에서 Gram d²와 직접 statevector d²를 비교했다. 최대 오차는 2.8×10⁻¹⁴이고,
  Gram 기울기와 tangent 기울기의 최대 오차는 2.6×10⁻¹⁴이다(허용 1e-10).
- 원래 실행의 witness 점(θ = 정수/17, x = 정수/19)에서 복사한 원본 interval 엔진(30자리)과 비교했다.
  d²와 기울기의 최대 오차는 1.2×10⁻¹⁴다. 이는 IR 의미론(큐비트 순서, 계수, CRZ ±1/2)을 다시 확인한 것이다.
- 수치 SVD(무작위 θ 4개, 인증 아님): 30개 모두 인증 랭크와 같다. 브래킷 설정에서는
  σ_r/σ₁ ≈ 3×10⁻⁵, σ_{r+1}/σ₁ ≈ 10⁻¹⁶이다.

## 예산 제한 32개 entry

원래 bounded Fourier pass(20 s, 쌍/항 예산)가 판정하지 못한 32개는 모두 Gram 경로에서 **INACTIVE**로 판정됐다.

- two-copy 불변 삭제 인증은 32/32다. 각 단계는 fresh-angle 기호 확인을 거쳤고, 설정당 0.01 s 미만이 걸렸다.
  메커니즘은 원래 structural closure 규칙(출력 수송 + metric/commutant)과 논리적으로 같다.
  독립 구현이지만 새로운 종류의 증거는 아니다.
- θ 전체를 기호로 둔 Gram 전개는 9/32를 판정했다.
  - Kyriienko n2 weight≤2 θ0–θ5: 0.02 s (bounded pass는 20 s 시간 초과).
  - Hubregtsen n2 L2 weight≤2 θ2, θ3: 2.7 s (bounded pass 20 s 시간 초과).
  - Kyriienko n2 XYZ θ2: 13.6 s (bounded pass는 0.17 s에 쌍 예산 초과).
- 나머지 23개는 기호 Gram 전개가 60 s 또는 60만 항 예산을 넘었다. YZ-CX L2와 n4 weight≤2,
  Hubregtsen n4 L2 Z, Kyriienko n4가 여기에 속한다. 이 경우에는 two-copy 불변 인증만 있다.
- 32개 모두 θ*에서 정확한 ∂_kG = 0 (mod p)이었고, 활성 증거와 충돌하지 않았다.
- 전체 30개 중 기호 Gram 전개가 완료된 설정은 16개다(원 bounded pass는 13개).

## 한계

- 산술 엔진(Python 정수, NumPy int64 mod p, SymPy Gröbner 약분)은 증명 보조기로 검증된 것이 아니다.
  정확 엔진, mod p 엔진, float statevector+DFT, 원본 interval 엔진이 서로 일치한다는 것은 구현 점검이지
  형식 증명이 아니다.
- U1 상한은 각 삭제와 가상 삽입이 K를 바꾸지 않는다는 구조적 논증, Euler chart 항등식, 그리고 θ* 한 점에서
  측정한 field 집합의 랭크(해석성으로 generic 랭크의 하한)에 기댄다. 기호 확인은 같은 Python/SymPy
  스택에서 수행했다.
- 하한은 설정마다 θ* 한 점에서 계산했다. 랭크는 정확하지만 다른 점에서는 더 작을 수 있으며, 이는 generic 정의와 맞다.
- 원고의 30개 결론이 바뀐 것은 없다. 새로 얻은 것은 Kyriienko 3개 구간의 값(4, 10, 10)과 그 인증이다.
  게이트·shot·학습 시간 절감은 측정하지 않았다. 이론 문서는 별도로 작성 중이며, 여기서는 운영 핵심만 구현했다.
- 첫 전체 실행(dry run)은 설정별 결과를 모두 낸 뒤 `summary` 단계의 이름 오류로 멈췄다. 그 설정별 결과도
  최종과 같았다. 이후 오류를 고치고 독립 기호 확인(삭제 단계와 가상 field)을 추가해 최종 실행을 새로 했다.
  dry run 결과는 결과 폴더에서 제외했다.

## 파일

- `fourier_gram.py`: IR 해석, Laurent 엔진(mod p/정확), 수치 엔진, 인증, 기호 two-copy 전개.
- `run_experiment.py`: 30개 실행, JSON, provenance, manifest.
- `test_fourier_gram.py`: 단위 테스트 11개. Hubregtsen G = diag(β₀,β₁)와 r=2, d² Gram=statevector,
  비활성 기호의 ∂_kG ≡ 0, 엔진 일치, Euler field 닫힌 형태, 음성 대조를 포함한다.
- `run.sh`: `test | full | brackets`.
- `results/20261002T044925Z/`: 설정별 JSON 30개, `summary.json`, `environment.json`, `provenance.json`, `manifest.json`.
