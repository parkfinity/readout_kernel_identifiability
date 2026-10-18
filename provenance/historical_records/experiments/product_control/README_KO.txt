고전적 product-feature 대조 실험

목적: v24의 잘못된 해석을 검증하고 수정한다. 이 Hubregtsen 회로에서는
정확한 ZZ 평균이 local-Z 평균의 곱이다. 추가 측정 정보가 반드시 필요하다는
예제가 아니라, 원래 네 feature의 고정 quadratic metric 한계를 비선형
feature lift로 넘는 예제이다. rank 2 / 3 / 4의 증명은 유지된다.

실행 범위
- 기존 v2의 3 panels x 5 held-out points x 4 directions x 6 arms x 4 budgets
- 1,440 paired comparison records, 각 256회 Monte Carlo
- 기존 metric, 방향, direct-parity contrast 고정; 같은 bitstring 표본 재사용
- 새 metric 선택, 새로운 외부 데이터셋, QPU 실행, 분류 정확도 실험 아님

결과
- 이상적 product feature 일치 최대 오차 1.78e-15
- product Z 확률 분해 일치 오차 1.67e-15
- 기존 Linux direct samples 재현 최대 차이 5.57e-14
- product estimator 분산 / direct estimator 분산의 panel별 중앙값:
  4096 shots에서 0.894760, 0.829430, 0.869455 (기존 noise-aware arm)
  즉 10.52%, 17.06%, 13.05% 낮은 분산. shot수 절감이나 학습성능 향상 아님.
- 위 수치는 covariance를 이용한 정확한 모델 분산의 비율이며, 256회
  Monte Carlo 표본분산만으로 계산한 효과 추정이나 신뢰구간이 아니다.
- product empirical/exact variance ratio: 중앙값 0.99023,
  5~95 percentile [0.86877, 1.11975]. 독립적인 1,440개 발견이 아니다.
- 4개 단위 테스트 통과. N=2, 4qubit 256개 표본열을 전수 열거하여
  unbiasedness, 전체 covariance, 조건부 기대값을 확인했다.

해석
제품분포에서는 직접 parity 추정량을 주변 count에 조건부 평균하면 곱 추정량이
된다. 따라서 같은 고정 metric/contrast에서 분산이 증가하지 않는 이유는
고전적인 조건부 기대값에 의한 분산 감소다. 새로운 양자 우월성이 아니다.
상관된 Z 분포나 상관 잡음에서는 이 대체가 일반적으로 유효하지 않다.
일반 grouped family의 최소 feature 차수/개수는 이번에 증명하지 않았다.

실행 방법 (bundle을 푼 뒤)
기존의 numpy 1.24.4, sympy 1.13.3 환경 사용 가능.
cd readout_product_baseline_v1
python3 -m unittest -v test_product
python3 run.py

bundle 안의 readout_metric_shot_design_v2 및
readout_metric_shot_v2_server_audit 폴더와 상대 위치를 유지한다.
results에는 이번 로컬 결과가 들어 있다. 재실행하면 같은 이름의 결과를
갱신하므로 원본 보존을 원하면 bundle을 별도 폴더에 다시 풀어 실행한다.
이번 product arm은 macOS 로컬 실행이며 서버 재현을 주장하지 않는다.
