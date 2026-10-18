readout_metric_shot_design_v2: 같은 shot 예산에서의 식별성 복원 검증

목적
v1의 정확한 rank 복원이 유한-shot에서 안정적인 감도로 이어지는지 확인.
3개 회로 계열, 4개 readout 설정(하나는 같은 Hubregtsen의 Z+ZZ 확장),
3 seed-panel, panel마다 설계 theta 3개/평가 theta 5개.
비교 6개: 등방, 고정 비등방 대각, 무작위 대각, 무작위 dense,
rank만 고려한 선택, shot noise를 고려한 선택. 선택 후보는 모두 동일.
추가 영상/분류 데이터, GPU, QPU 계정이나 유료 작업은 필요 없다.

서버 사용법
ZIP을 /workspace에 업로드한 뒤 아래 명령을 실행한다.
명령 앞 프롬프트나 Markdown의 ```는 터미널에 입력하지 않는다.

cd /workspace
python3 -m zipfile -e readout_metric_shot_design_v2_bundle.zip .
cd readout_metric_shot_design_v2
source /workspace/.venv_readout_metric/bin/activate
python3 -m pip install -r requirements_py38.txt
bash run.sh smoke
bash run.sh pilot
bash run.sh all
python3 -m zipfile -c readout_metric_shot_v2_server_results.zip results

기존 .venv_readout_metric이 없다면 source 전에 다음을 실행한다.
python3 -m venv /workspace/.venv_readout_metric

smoke: 10개 단위 테스트 및 고정 설계 hash 검사.
pilot: 첫 panel/첫 평가 theta, 4개 설정, 1024/4096 shots, 64회 반복.
all: 3 panel x 4 설정 x 5 theta = 결과 60개,
256/1024/4096/16384 shots, 256회 반복. 각 theta에서 공통 target의 모든
canonical 방향을 평가한다. 각 결과마다 진행 상황이 출력된다.
각 실행은 별도 UTC 폴더를 생성하고 기존 결과를 덮어쓰지 않는다.
이 패키지에는 로컬 결과 results/를 넣지 않아 서버 실행과 혼동하지 않는다.
로컬 파일럿 점검 기록은 LOCAL_QA.json에 별도로 제공한다.
전체 실행 시간은 장비에 따라 다르며 pilot 시간은 full 시간을 보장하지 않는다.

완료 확인 및 반환
마지막에 complete: true, analysis의 passed: true가 출력되어야 한다.
summary.json과 analysis.json이 결과 폴더에 생성된다.
readout_metric_shot_v2_server_results.zip을 전달하면 원고 업데이트 전에
재현성/통계/실패 사례를 대조한다. 압축 뒤 프롬프트가 보이면 완료 상태다.

주의
prepare.py는 개발용이다. 서버에서는 실행하지 않는다. plan.json에 고정된
설계와 방향을 그대로 사용하므로 v1의 SVD 부호 차이를 피한다.
run.sh는 plan/입력의 hash가 다르면 중단한다. 원본 파일을 수정하지 않는다.
metric과 측정값은 synthetic ideal model을 사용해 설계된다. 이 실험은
shot만으로 학습되는 실용 최적화기 또는 실제 장비 검증이 아니다.
rank/감도는 수치 진단이며 v1의 exact certificate와 구분한다.
Monte Carlo contrast는 모든 방법에 동일하게 허용한 ideal-model oracle다.
SNR는 검정력/분류 정확도/동일 정밀도 shot 절감을 뜻하지 않는다.
보이지 않던 방향의 복원과 전체 rank 증가를 구분해 분석한다.

공동 측정 회계
local XYZ는 총 shots를 세 basis에 나눈다(4096 요청이면 실제 4092).
Hub Z와 Z+ZZ는 같은 ZZZZ basis/counts를 공유한다. feature마다 새 budget을
부여하지 않는다. 16쌍, 두 parameter state의 총비용은 방향당 64 x 실제
shots/input/state이다. 모든 방향/반복의 총비용과 혼동하지 않는다.
같은 counts를 재사용할 수 있다는 말은 동일 분산을 보장한다는 뜻이 아니다.

원고 반영 계획은 MANUSCRIPT_UPDATE_PLAN_KO.txt,
실험 정의와 해석 기준은 PROTOCOL.txt에 있다. 
