# JAAD Jaywalking Risk Recognition - Progress Handoff

이 문서는 VS Code를 끄고 다음 날 다시 작업할 때 바로 이어서 볼 수 있는 진행상황 정리 문서입니다.

## 1. 현재 목표

본 프로젝트의 목표는 JAAD 데이터셋을 이용해 보행자의 무단횡단 또는 도로 진입 위험을 예측하는 baseline 실험 파이프라인을 구성하는 것입니다.

현재 연구 질문은 다음과 같습니다.

- 보행자 bbox 정보만 사용할 때보다 pose feature가 성능을 개선하는가?
- pose feature와 road relation feature를 함께 사용하면 jaywalking risk recognition에 도움이 되는가?
- RandomForest baseline과 LSTM sequence model의 성능 차이는 어떤가?

## 2. 현재까지 완료된 작업

### JAAD 데이터 준비

- JAAD video clips 다운로드 스크립트 작성 및 개선
  - `scripts/download_jaad_clips.ps1`
  - 다운로드 중단/재시도 상황을 고려해 기존 zip 사용 옵션을 추가했습니다.
- JAAD 2.0 annotation repository 클론
  - 경로: `data/raw/JAAD_annotations`
  - 주요 폴더:
    - `annotations`
    - `annotations_attributes`
    - `annotations_appearance`
    - `annotations_traffic`
    - `annotations_vehicle`
    - `split_ids`
- 전체 JAAD 영상 346개 기준 feature 생성 완료
  - 실패 영상 수: 0
  - 실패 로그: `outputs/logs/jaad_failed_videos.txt`

### JAAD official split 적용

- `data/raw/JAAD_annotations/split_ids`의 official split 파일을 읽도록 구현했습니다.
- `jaad_features.csv`에 `split` 컬럼을 추가했습니다.
- train/val/test 기준으로 학습과 평가가 동작하도록 수정했습니다.
- random split은 옵션으로 남기고, 기본값은 JAAD official split입니다.
- 실행 시 train/test/val 간 `video_id` overlap 검증 로그가 출력됩니다.

### Feature generation pipeline

- JAAD annotation parser 추가
  - `src/jaad_loader.py`
- feature 생성 로직 개선
  - `src/dataset_builder.py`
  - `src/feature_extractor.py`
- 전체 영상 처리 중 특정 영상이 실패해도 다음 영상으로 계속 진행하도록 수정했습니다.
- 전체 JAAD feature CSV 생성 경로:
  - `data/features/jaad_features.csv`

### Data quality report

- 데이터 품질 리포트 생성 기능 추가
  - `src/data_quality.py`
- 출력 파일:
  - `outputs/reports/jaad_data_quality_report.csv`
  - `outputs/reports/jaad_data_quality_summary.txt`
- 분석 항목:
  - 전체 row 수
  - 처리된 video 수
  - pedestrian_id 수
  - split별 row 수
  - split별 video 수
  - label 분포
  - video별 row 수
  - 결측치 개수
  - 주요 feature의 min/max/mean/std
  - train/val/test 간 video_id overlap 여부

### Training validation

- RandomForest와 LSTM 학습 전 검증 기능을 추가했습니다.
- 확인 항목:
  - label이 한쪽 클래스만 있는지
  - train/test에 동일 `video_id`가 섞였는지
  - NaN 또는 inf 값이 있는지
  - feature column이 누락되었는지
- 심각한 문제가 있으면 학습을 중단하고, 경미한 문제는 warning을 출력합니다.

### RandomForest baseline automation

- baseline 자동 실행 모드 추가
  - `src/baseline_runner.py`
- 실행 모드:
  - `run-baselines`
- feature set:
  - `bbox_only`
  - `pose_only`
  - `road_relation_only`
  - `pose_road_relation`
- 결과 저장:
  - `outputs/results/baseline_results.csv`

### LSTM experiment

- LSTM 학습 전 sequence 통계를 출력하도록 수정했습니다.
- 출력 항목:
  - train sequence 수
  - test sequence 수
  - sequence별 label 분포
  - 너무 짧아서 제외된 sequence 수
- LSTM 결과 저장:
  - `outputs/results/lstm_results.csv`

### Experiment summary

- RandomForest baseline 결과와 LSTM 결과를 함께 비교할 수 있도록 summary 생성 기능을 추가했습니다.
  - `src/experiment_summary.py`
- 결과 저장:
  - `outputs/results/experiment_summary.csv`

### YOLOv8 pose 적용

- 기존 dummy pose 외에 YOLOv8 pose backend를 추가했습니다.
- `configs/default.yaml`에 pose 설정 추가:

```yaml
pose:
  backend: yolo
  model_path: models/yolo/yolov8n-pose.pt
  inference_mode: bbox
```

- YOLO pose extractor 수정:
  - `src/pose_extractor.py`
- 로컬 weights 파일만 사용하도록 했습니다.
- 인터넷 자동 다운로드는 코드에서 제거했습니다.
- weights가 없거나 YOLO 로드가 실패하면 fallback pose를 사용합니다.
- 실행 시 pose backend, model path, inference mode가 로그로 출력됩니다.
- 현재 사용한 weights:
  - `models/yolo/yolov8n-pose.pt`
- 이 파일은 용량이 크기 때문에 Git에는 올리지 않습니다.

### GPU 환경 확인

현재 사용자가 설치 및 확인한 PyTorch CUDA 환경:

```text
torch: 2.5.1+cu121
CUDA available: True
GPU: NVIDIA GeForce GTX 1650 SUPER
```

YOLO pose 기반 전체 JAAD feature 생성이 GPU 환경에서 완료되었습니다.

## 3. 최신 실험 실행 기록

### 1. 전체 JAAD feature 생성

```powershell
.\.venv\Scripts\python.exe main.py --mode jaad-features
```

결과:

```text
JAAD videos: 346/346
failed videos: 0
feature CSV: data/features/jaad_features.csv
quality reports saved
```

### 2. 데이터 품질 리포트 생성

```powershell
.\.venv\Scripts\python.exe main.py --mode quality-report --csv-path data\features\jaad_features.csv
```

결과:

```text
outputs/reports/jaad_data_quality_report.csv
outputs/reports/jaad_data_quality_summary.txt
```

### 3. RandomForest baseline 실행

```powershell
.\.venv\Scripts\python.exe main.py --mode run-baselines --csv-path data\features\jaad_features.csv
```

결과:

```text
outputs/results/baseline_results.csv
outputs/results/experiment_summary.csv
```

### 4. LSTM 학습 실행

```powershell
.\.venv\Scripts\python.exe main.py --mode train-lstm --csv-path data\features\jaad_features.csv --feature-set pose_road_relation
```

결과:

```text
outputs/results/lstm_results.csv
outputs/results/experiment_summary.csv
```

## 4. 최신 결과 요약

### Data quality

```text
total rows: 26,615
processed videos: 320
pedestrians: 686
train rows: 13,269
val rows: 2,124
test rows: 11,222
label 0: 11,626
label 1: 14,989
train/test overlap: 0
train/val overlap: 0
val/test overlap: 0
```

주의:

- 전체 영상 입력은 346개였고 feature 생성 실패는 0개입니다.
- feature row가 생성된 video는 320개입니다.
- 나머지 영상은 annotation 조건 또는 추출 조건상 최종 feature row가 없을 수 있습니다.

### RandomForest baseline

| feature_set | model | f1 |
|---|---:|---:|
| bbox_only | RandomForest | 0.6560 |
| pose_only | RandomForest | 0.7108 |
| road_relation_only | RandomForest | 0.7200 |
| pose_road_relation | RandomForest | 0.7102 |

### LSTM result

| feature_set | model | accuracy | precision | recall | f1 |
|---|---|---:|---:|---:|---:|
| pose_road_relation | LSTM | 0.6601 | 0.7647 | 0.6404 | 0.6971 |

LSTM sequence 통계:

```text
train sequences: 12,659
test sequences: 9,184
train excluded short tracks: 12
test excluded short tracks: 5
```

## 5. 주요 파일 설명

| 파일 | 설명 |
|---|---|
| `README.md` | 논문 스타일 프로젝트 설명, 실행 방법, 결과 정리 |
| `PROGRESS.md` | 현재 문서. 내일 이어서 작업하기 위한 인수인계 문서 |
| `configs/default.yaml` | 기본 설정 파일. JAAD, pose, training, feature set 설정 포함 |
| `main.py` | 실행 entrypoint. mode별 기능 실행 |
| `src/jaad_loader.py` | JAAD annotation 및 split 파일 로더 |
| `src/dataset_builder.py` | JAAD feature CSV 생성 파이프라인 |
| `src/pose_extractor.py` | dummy 또는 YOLOv8 pose feature 추출 |
| `src/data_quality.py` | 데이터 품질 리포트 생성 |
| `src/baseline_runner.py` | RandomForest baseline 자동 실행 |
| `src/train_random_forest.py` | RandomForest 단일 학습 |
| `src/train_lstm.py` | LSTM sequence 학습 |
| `src/experiment_summary.py` | RF/LSTM 결과 통합 summary 생성 |
| `scripts/download_jaad_clips.ps1` | JAAD clips 다운로드 스크립트 |
| `scripts/create_progress_report_docx.py` | 진행상황 Word 문서 생성 스크립트 |

## 6. Git 상태

최근 GitHub에 push한 커밋:

```text
7ce4a16 Add JAAD full experiment pipeline and YOLO pose support
```

원격 저장소:

```text
https://github.com/leoa4238/Researchai.git
```

주의:

- `models/yolo/yolov8n-pose.pt`는 Git에 올리지 않습니다.
- `data/raw`, `data/features`, `outputs/results`, `outputs/logs`, `outputs/reports`의 실제 결과 파일은 기본적으로 Git에 올리지 않습니다.
- 대신 재현 가능한 실행 명령어와 코드만 Git에 올립니다.

## 7. 내일 바로 실행할 명령어

프로젝트 루트로 이동:

```powershell
cd D:\reserch\jaywalking-risk-recognition
```

가상환경 Python 확인:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA not available')"
```

YOLO weights 파일 확인:

```powershell
Test-Path models\yolo\yolov8n-pose.pt
```

전체 feature 재생성:

```powershell
.\.venv\Scripts\python.exe main.py --mode jaad-features
```

품질 리포트 생성:

```powershell
.\.venv\Scripts\python.exe main.py --mode quality-report --csv-path data\features\jaad_features.csv
```

RandomForest baseline 실행:

```powershell
.\.venv\Scripts\python.exe main.py --mode run-baselines --csv-path data\features\jaad_features.csv
```

LSTM 학습:

```powershell
.\.venv\Scripts\python.exe main.py --mode train-lstm --csv-path data\features\jaad_features.csv --feature-set pose_road_relation
```

## 8. 다음에 하면 좋은 작업

1. YOLO pose 결과를 더 안정적으로 만들기
   - bbox crop 방식과 full frame 방식 비교
   - keypoint confidence threshold 조정
   - fallback 사용 비율 로그 추가

2. road relation feature 개선
   - 현재 road relation은 아직 단순화된 feature입니다.
   - segmentation 기반 도로 영역 추출을 붙이면 연구 기여가 더 명확해집니다.

3. LSTM 모델 개선
   - sequence length, hidden size, learning rate 조정
   - validation split을 이용한 early stopping 추가
   - confusion matrix 저장 추가

4. ST-GCN 또는 pose graph 모델 추가
   - YOLO keypoints를 graph 구조로 사용
   - pose temporal dynamics를 더 직접적으로 모델링

5. 논문용 결과 정리
   - `baseline_results.csv`
   - `lstm_results.csv`
   - `experiment_summary.csv`
   - README Results 표 업데이트

## 9. 현재 주의할 점

- 현재 결과는 YOLOv8n pose 기반이지만 road relation은 아직 완전한 segmentation 기반이 아닙니다.
- README의 Limitations에 dummy/단순 road relation 한계를 명시했습니다.
- `jaad_features.csv`는 전체 실험의 핵심 입력 파일이지만 Git에는 올리지 않습니다.
- VS Code를 닫고 다시 시작해도 이 문서와 README를 보면 현재 상태를 복원할 수 있습니다.
