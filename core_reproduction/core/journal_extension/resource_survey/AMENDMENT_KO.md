# 상수 커널의 미정의 alignment 처리

첫 로컬 실행에서 ParamZFeatureMap의 local Z readout 두 슬롯은 centered kernel alignment의 분모가 0이어서 검증 예외가 발생했다. 원 실행 결과를 results에 그대로 보존했다. 이를 인증 실패나 새 절감 사례로 재분류하지 않는다.

두 번째 실행에서는 모든 슬롯에 kernel Jacobian의 유한차분 검사와 공유-angle 특징 Jacobian의 analytic 비교를 추가했다. Alignment가 정의되지 않는 슬롯에서는 objective gradient 및 두-step 학습을 실행하지 않고 PASS_KERNEL_ONLY_ALIGNMENT_UNDEFINED로 따로 집계한다. 해당 슬롯의 커널·kernel derivative·고정 파라미터 ridge 예측 검사는 유지한다. 조사 분모, 소스, constructor, seed, readout, 기준선, 인증 및 허용값을 유리한 결과를 찾도록 바꾸지 않았다. 이는 첫 실행 후 작성한 명시적인 처리 수정이며 사전등록이라고 부르지 않는다.
