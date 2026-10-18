# 로컬 검증 범위

2026-10-01. 최종 수치 설계는 `20261001_boundary_fused_v3`입니다. 서버 실행용 패키지를 준비하고 로컬에서 검증했으며, Python 3.8 서버 full 실험은 아직 실행하지 않았습니다. 로컬 환경은 Python 3.9.6, NumPy 1.24.4, Qiskit 1.2.4, SymPy 1.14.0입니다. 서버 요구사항의 SymPy 1.13.3과 다르므로 서버 재검증이 필요합니다.

## 최종 설계 pilot

실행: `pilot_20261001T041204_099569Z`.

- 25개 단위 테스트 통과.
- 전체 12개 설정의 원본 source·IR·세 미분 backend·source finite difference 사전 검증 통과.
- Ionosphere seed42, 64 train + 64 test, 4 optimizer steps, 1 timing repetition.
- 36 paired cold blocks, 7 arms씩 총 252회 학습.
- 6 paired 재사용 blocks, 양 군 5회씩 총 60회 학습. H=1·2·5의 누적 시간을 실제 측정.
- 저장된 312개 Gram 배열 참조 및 paired 경로 검증 통과. 정확히 같은 배열은 lossless content-addressed 방식으로 중복 저장하지 않음.
- 별도 scalar source 경로로 cold reference 12개와 재사용 fit 10개를 재계산.

| 재계산 항목 | 최대 절대 오차 |
|---|---:|
| Kernel | 1.9984×10⁻¹⁵ |
| Gradient | 4.3077×10⁻¹³ |
| Objective | 3.0930×10⁻¹⁵ |
| Prediction | 3.9080×10⁻¹⁴ |

이 수치는 보존 검증입니다. 독립 연구자의 proof review, 새로운 문헌 prevalence, 전체 sQUlearn runtime 검증, 서버 시간 재현이 아닙니다. 실제 시간은 원자료에 모두 남겼지만 단일 dataset/seed/repetition의 pilot 절감률을 논문 주 효과로 사용하지 않습니다.

## 개발 중 변경 기록

최초 unit fixture는 local readout이 상수인 Bell-phase family여서 centered alignment가 정의되지 않았습니다. 구현은 오류를 냈으며, zero-active 테스트만 다른 데이터 의존 family로 수정했습니다. 퇴화 objective 보호 조건은 유지했습니다.

첫 smoke 및 첫 pilot은 local-tail 대비 full transport였습니다. 이후 genuine later-data crossing을 구분하기 위해 boundary 기준선을 추가했습니다. 다음에는 양 군 모두 adjacent same-axis rotation fusion을 적용해 IR 분해로 인한 인위적 비용 차이를 제거했습니다. 이전 두 pilot은 삭제하지 않았으며 최종 설계와 합산하지 않습니다. 최종 pilot 이후에는 release hash 검사 및 vendor/frozen-input snapshot 등 배포·증거 보존 기능을 보완했고 수치 알고리즘·평가 대상·기준선은 바꾸지 않았습니다.

## Full 실행 판정

Full은 3 datasets×5 seeds, 12 circuit configurations, 3 backends, 7 arms, 5 repetitions의 cold 실험과 seed42 재사용 부분집합입니다. 완료 기준은 2,700 cold blocks / 18,900 cold fits 및 90 reuse blocks / 3,600 reuse fits, 그리고 summary와 audit의 passed=true입니다. 큰 출력과 실행 시간이 생길 수 있습니다. 실패·음성 결과도 보존합니다.
