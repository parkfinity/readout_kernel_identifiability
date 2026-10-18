readout_metric_design_v1: 식별 가능성 복원 설계 파일럿

실험은 이미 로컬에서 실행했다. RESULTS_KO.txt를 먼저 읽는다.
PROTOCOL.txt는 최초 파일럿 범위, FOLLOWUP_PROTOCOL.txt는 결과를 본 뒤
추가한 강한 기준선/설명용 사례의 범위다. 후속 분석을 사전 등록으로 꾸미지 않는다.

필요 환경: CPU, Python 3.8 이상. 작은 상태벡터(2/4큐비트) 실험이다.
추가 데이터셋 다운로드, GPU, Qiskit, QPU 계정/비용이 필요하지 않다.
frozen_inputs/와 vendor/만으로 실행하며 상위 연구 폴더에 의존하지 않는다.
기본 실험 입력은 합성 연속 입력이다. Moons/Wine/영상 성능 실험이 아니다.

서버 설치 예시 (ZIP 파일을 /workspace에 업로드한 뒤):

cd /workspace
python3 -m zipfile -e readout_metric_design_v1_bundle.zip .
cd readout_metric_design_v1
python3 -m venv /workspace/.venv_readout_metric
source /workspace/.venv_readout_metric/bin/activate
python3 -m pip install -r requirements_py38.txt
bash run.sh smoke
bash run.sh all
python3 -m zipfile -c readout_metric_server_results.zip results

복사할 때 명령줄 앞의 $ 또는 Markdown의 ```를 터미널에 넣지 않는다.
all은 8개 테스트, 수치 survey, 정확 인증, 독립 소행렬식 재검사,
강한 기준선, Z 상관관계 추가 대조를 순서대로 실행한다.
새 UTC 시간 폴더를 만들며, 동봉한 로컬 결과 두 세트는 덮어쓰지 않는다.
환경에 따라 시간이 달라진다. 수치 파일럿은 로컬 CPU에서 약 4초였고
추가 정확 인증과 파일 저장 시간이 따로 있다. 서버 시간을 보장하지 않는다.

서버에서 돌릴 핵심 파일:
design.py, experiment.py, exact_controls.py, verify_certificates.py,
followup.py, correlation_control.py, test_design.py, run.sh.
summarize_and_package.py는 로컬 유지관리 도구다. 기존 연구 폴더의 원본
엔진과 대조하기 때문에 서버에서 실행할 필요가 없으며 all에 포함되지 않는다.

결과 확인:
- summary.json: 수치 실험 범위와 독립 상태벡터 오차.
- metric_summary.json: 30개 설정과 metric별 모든 held-out 진단.
- readout_summary.json: 10개 회로, 1/2/3 basis 예산, 무작위 포함 모든 결과.
- exact_controls/summary.json: 20개 full-rank 정확 인증.
- certificate_recheck.json: SymPy 소행렬식 재검사, 3개 완전 유리수 증인.
- correlation_control.json: local Z rank 2 / dense metric 3 / Z+ZZ rank 4.
- strong_baseline_followup/: all-X/Y/Z 및 feature variance 기준선, shot 비교.
- finite_shots.json: 실제 하드웨어가 아닌 이상적 joint-shot 시뮬레이션.

수치 rank는 exact generic rank와 구분한다. 무작위 설계와 greedy가
동일 rank인 경우도 모두 남겼다. OS/BLAS에 따른 거의 동점의 basis 선택은
달라질 수 있으므로, bitwise 동일성 대신 해당 설계의 정확 인증 및 지표를
확인한다. seed와 데이터/파라미터 분리는 고정되어 있다.

자료 공개는 수행하지 않았다. 이 ZIP은 검토/재현용이며 공개 저장소 및
라이선스 결정은 저자가 별도로 한다. 
