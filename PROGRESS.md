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

### YOLO pose 품질 분석 추가

- detection별 YOLO keypoint 추출 성공 여부와 dummy fallback 여부를 기록하도록 추가했습니다.
  - 수정 파일: `src/pose_extractor.py`
- JAAD feature 생성 시 `video_id`, `frame_id`, `pedestrian_id`, inference mode, success/fallback 여부를 추적합니다.
- 전체 detection 수, success 수, fallback 수, success rate를 summary로 저장합니다.
- video_id별 success rate를 CSV로 저장합니다.
- bbox crop mode와 full frame mode를 각각 실행해 비교할 수 있도록 CLI 옵션을 추가했습니다.
  - `--pose-inference-mode bbox`
  - `--pose-inference-mode full_frame`
- 결과 저장:
  - `outputs/reports/pose_detection_report.csv`
  - `outputs/reports/pose_detection_summary.txt`
  - `outputs/reports/pose_detection_events.csv`
  - `outputs/reports/pose_detection_report_yolo_bbox.csv`
  - `outputs/reports/pose_detection_summary_yolo_bbox.txt`
  - `outputs/reports/pose_detection_events_yolo_bbox.csv`
  - `outputs/reports/pose_detection_report_yolo_full_frame.csv`
  - `outputs/reports/pose_detection_summary_yolo_full_frame.txt`
  - `outputs/reports/pose_detection_events_yolo_full_frame.csv`
- bbox/full_frame별 feature CSV와 baseline 결과를 구분 저장하도록 했습니다.
  - `data/features/jaad_features_yolo_bbox.csv`
  - `data/features/jaad_features_yolo_full_frame.csv`
  - `outputs/results/baseline_results_yolo_bbox.csv`
  - `outputs/results/baseline_results_yolo_full_frame.csv`

### JAAD traffic annotation 분석 및 signal feature 추가

- `data/raw/JAAD_annotations/annotations_traffic` XML 구조를 분석하는 기능을 추가했습니다.
  - 실행 모드: `traffic-analysis`
  - 구현 파일: `src/traffic_annotations.py`
- 분석 결과:
  - root tag: `traffic_scene`
  - frame attribute: `id`, `ped_crossing`, `ped_sign`, `stop_sign`, `traffic_light`
  - `traffic_light` 값:
    - `n/a`: 77,599
    - `green`: 2,263
    - `red`: 2,170
  - `yellow`는 현재 JAAD traffic annotation에서 관측되지 않았습니다.
- red/green 상태 구분은 가능하지만, traffic light가 차량용인지 보행자용인지는 XML만으로 구분할 수 없습니다.
- 분석 리포트 저장:
  - `outputs/reports/traffic_annotation_analysis.txt`
  - `outputs/reports/traffic_annotation_values.csv`
- JAAD feature CSV에 아래 컬럼을 추가했습니다.
  - `traffic_light_present`
  - `traffic_light_state`
  - `traffic_light_state_code`
  - `risk_label`
- `traffic_light_state_code` encoding:
  - `unknown=0`
  - `red=1`
  - `yellow=2`
  - `green=3`
- `risk_label` 정의:
  - `label == 1` 그리고 `traffic_light_state == red`이면 `risk_label = 1`
  - `traffic_light_state in [green, yellow]`이면 `risk_label = 0`
  - 그 외에는 `risk_label = -1`
- `risk_label = -1` row는 risk 학습 시 제외합니다.
- 새 feature set:
  - `pose_road_signal`
- baseline target 선택 옵션을 추가했습니다.
  - `--target-column label`
  - `--target-column risk_label`
- baseline 결과 저장:
  - `outputs/results/baseline_results_crossing.csv`
  - `outputs/results/baseline_results_risk.csv`
- README에 분석 화면 요약 이미지를 추가했습니다.
  - `docs/images/analysis_workflow_capture.png`
  - `docs/images/analysis_process_pipeline.png`
  - `docs/images/feature_row_process_capture.png`
  - `docs/images/traffic_annotation_analysis_capture.png`
  - `docs/images/pose_quality_capture.png`
  - `docs/images/baseline_results_capture.png`
- 이미지 생성 스크립트:
  - `scripts/create_analysis_screenshots.py`
- 논문/보고서용 근거 정리 문서:
  - `docs/paper_method_evidence.md`

### Road segmentation backend abstraction 추가

- 기존 dummy road mask 생성 로직은 `src/road_segmenter.py`에 있었습니다.
  - 기존 방식: frame 하단 45%를 road, 그 위 일부를 sidewalk로 가정
  - 이 mask를 이용해 `distance_to_road`, `foot_on_road`, `center_on_road`, `approach_rate`를 계산했습니다.
- `RoadSegmenter`를 backend abstraction 구조로 확장했습니다.
  - `dummy`
  - `deeplabv3`
  - `segformer`
  - `yolo_seg`
- 실제 segmentation model이 없거나 로드/추론에 실패하면 기존 dummy mask로 fallback됩니다.
- `main.py`에 segmentation CLI 옵션을 추가했습니다.
  - `--segmentation-backend dummy|deeplabv3|segformer|yolo_seg|yolo-seg`
  - `--segmentation-model-path <path>`
- `configs/default.yaml`에 segmentation 설정을 확장했습니다.
  - `segmentation.model_path`
  - `segmentation.device`
  - `segmentation.num_classes`
  - `segmentation.road_class_ids`
  - `segmentation.sidewalk_class_ids`
  - `segmentation.road_class_names`
  - `segmentation.sidewalk_class_names`
- feature CSV에 segmentation metadata 컬럼을 추가했습니다.
  - `segmentation_backend`
  - `segmentation_backend_requested`
  - `segmentation_source`
  - `segmentation_fallback`
  - `road_pixel_ratio`
- road segmentation 품질 리포트를 저장합니다.
  - `outputs/reports/road_segmentation_report.csv`
  - `outputs/reports/road_segmentation_summary.txt`
  - `outputs/reports/road_segmentation_report_<backend>.csv`
  - `outputs/reports/road_segmentation_summary_<backend>.txt`
- backward compatibility:
  - 기본 backend는 `dummy`입니다.
  - 기존 feature set과 baseline 학습 컬럼은 그대로 유지됩니다.
  - segmentation metadata는 추가 컬럼이므로 기존 학습 로직을 깨지 않습니다.
- sanity check:
  - `--segmentation-backend dummy`로 `video_0002` 1개 영상 feature 생성 성공
  - `--segmentation-backend yolo_seg`에서 model path가 없을 때 dummy fallback 성공
- yolo_seg 최소 예제 스크립트를 추가했습니다.
  - `scripts/run_road_segmentation_example.py`
  - checkpoint가 있으면 1프레임 road mask preview를 생성합니다.
  - 현재 `models/segmentation/`에는 checkpoint가 없으므로 dummy fallback으로 기록됩니다.
- road relation feature 분포 비교 스크립트를 추가했습니다.
  - `scripts/compare_road_relation_features.py`
  - 출력: `outputs/reports/road_relation_comparison.csv`
- `--limit-videos 5` sanity check:
  - dummy CSV: `data/features/test_jaad_features_road_dummy_5.csv`
  - yolo_seg 요청 CSV: `data/features/test_jaad_features_road_yolo_seg_5.csv`
  - 현재 checkpoint가 없으므로 yolo_seg 요청 결과는 `segmentation_backend=dummy`, `segmentation_backend_requested=yolo_seg`, `segmentation_fallback=1`로 기록되었습니다.
  - `outputs/reports/road_segmentation_summary_yolo_seg.txt`의 fallback ratio는 `1.0000`입니다.
  - 평균 road pixel ratio는 dummy/fallback 모두 `0.4500`입니다.

### GPU 환경 확인

현재 사용자가 설치 및 확인한 PyTorch CUDA 환경:

```text
torch: 2.5.1+cu121
CUDA available: True
GPU: NVIDIA GeForce GTX 1650 SUPER
```

YOLO pose 기반 전체 JAAD feature 생성이 GPU 환경에서 완료되었습니다.

## 3. 최신 실험 실행 기록

### 1. bbox mode 전체 JAAD feature 생성

```powershell
.\.venv\Scripts\python.exe main.py --mode jaad-features --pose-inference-mode bbox
```

결과:

```text
JAAD videos: 346/346
failed videos: 0
feature CSV: data/features/jaad_features_yolo_bbox.csv
pose report: outputs/reports/pose_detection_summary_yolo_bbox.txt
quality reports saved
```

### 2. bbox mode RandomForest baseline 실행

```powershell
.\.venv\Scripts\python.exe main.py --mode run-baselines --pose-inference-mode bbox
```

결과:

```text
outputs/results/baseline_results_yolo_bbox.csv
```

### 3. full_frame mode 전체 JAAD feature 생성

```powershell
.\.venv\Scripts\python.exe main.py --mode jaad-features --pose-inference-mode full_frame
```

결과:

```text
JAAD videos: 346/346
failed videos: 0
feature CSV: data/features/jaad_features_yolo_full_frame.csv
pose report: outputs/reports/pose_detection_summary_yolo_full_frame.txt
```

### 4. full_frame mode RandomForest baseline 실행

```powershell
.\.venv\Scripts\python.exe main.py --mode run-baselines --pose-inference-mode full_frame
```

결과:

```text
outputs/results/baseline_results_yolo_full_frame.csv
```

### 5. LSTM 학습 실행

```powershell
.\.venv\Scripts\python.exe main.py --mode train-lstm --csv-path data\features\jaad_features.csv --feature-set pose_road_relation
```

결과:

```text
outputs/results/lstm_results.csv
outputs/results/experiment_summary.csv
```

### 6. traffic annotation 분석 실행

```powershell
.\.venv\Scripts\python.exe main.py --mode traffic-analysis
```

결과:

```text
outputs/reports/traffic_annotation_analysis.txt
outputs/reports/traffic_annotation_values.csv
```

### 7. traffic feature sanity check 실행

```powershell
.\.venv\Scripts\python.exe main.py --mode jaad-features --pose-inference-mode bbox --limit-videos 20 --output-csv data\features\test_jaad_features_signal_20.csv
.\.venv\Scripts\python.exe main.py --mode run-baselines --csv-path data\features\test_jaad_features_signal_20.csv --target-column label --baseline-output outputs\results\test_baseline_results_crossing_20.csv
.\.venv\Scripts\python.exe main.py --mode run-baselines --csv-path data\features\test_jaad_features_signal_20.csv --target-column risk_label --baseline-output outputs\results\test_baseline_results_risk_20.csv
```

sanity check 결과:

```text
20-video feature rows: 1,054
traffic_light_present=1 rows: 22
traffic_light_state values: red, unknown
risk_label=-1 rows: 1,032
risk_label trainable rows: 22
```

주의: 20개 영상 sanity check에서는 risk_label 학습 가능 row가 모두 class 1이라 risk baseline은 `skipped`로 저장되었습니다. 전체 JAAD 기준에서는 green frame이 포함되므로 risk baseline을 다시 실행해야 합니다.

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
traffic_light_present 0: 24,677
traffic_light_present 1: 1,938
traffic_light_state green: 904
traffic_light_state red: 1,034
traffic_light_state unknown: 24,677
risk_label -1: 25,145
risk_label 0: 904
risk_label 1: 566
risk_label trainable rows: 1,470
train/test overlap: 0
train/val overlap: 0
val/test overlap: 0
```

주의:

- 전체 영상 입력은 346개였고 feature 생성 실패는 0개입니다.
- feature row가 생성된 video는 320개입니다.
- 나머지 영상은 annotation 조건 또는 추출 조건상 최종 feature row가 없을 수 있습니다.

### RandomForest baseline

아래 결과는 bbox crop YOLO pose inference를 기본 결과로 사용한 RandomForest baseline입니다.

| feature_set | model | accuracy | precision | recall | f1 |
|---|---|---:|---:|---:|---:|
| bbox_only | RandomForest | 0.6043 | 0.6395 | 0.6733 | 0.6560 |
| pose_only | RandomForest | 0.6631 | 0.6847 | 0.7389 | 0.7108 |
| road_relation_only | RandomForest | 0.5653 | 0.5633 | 0.9975 | 0.7200 |
| pose_road_relation | RandomForest | 0.6626 | 0.6847 | 0.7376 | 0.7102 |
| pose_road_signal | RandomForest | 0.6794 | 0.7026 | 0.7419 | 0.7217 |

해석:

- `pose_road_signal`은 crossing classification에서 F1 0.7217로 가장 높았습니다.
- 기존 `pose_road_relation` F1 0.7102보다 상승했으므로, frame-level traffic context가 crossing classification에 추가 정보를 줄 가능성이 있습니다.
- 다만 traffic light는 frame-level context이며 차량용/보행자용 구분은 없습니다.

### Risk prediction baseline

`risk_label=-1` row를 제외하고 학습했습니다.

| feature_set | model | accuracy | precision | recall | f1 | train_rows | test_rows |
|---|---|---:|---:|---:|---:|---:|---:|
| bbox_only | RandomForest | 0.7623 | 0.9007 | 0.5620 | 0.6921 | 961 | 509 |
| pose_only | RandomForest | 0.7800 | 0.9924 | 0.5413 | 0.7005 | 961 | 509 |
| road_relation_only | RandomForest | 0.5246 | 0.0000 | 0.0000 | 0.0000 | 961 | 509 |
| pose_road_relation | RandomForest | 0.7721 | 0.9922 | 0.5248 | 0.6865 | 961 | 509 |
| pose_road_signal | RandomForest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 961 | 509 |

해석:

- risk prediction 전체 학습 가능 row는 1,470개입니다.
- `pose_road_signal`의 F1 1.0은 모델이 일반적인 위험 판단을 완전히 학습했다는 의미로 해석하면 안 됩니다.
- 현재 `risk_label`은 `traffic_light_state`로부터 정의되고, `pose_road_signal`에는 `traffic_light_state_code`가 포함되어 있어 정답 규칙을 feature로 제공한 상태에 가깝습니다.
- 따라서 이 결과는 pipeline sanity check로는 유효하지만, 논문/보고서에서는 target-feature dependency 한계로 명시해야 합니다.

### Signal presence split baseline

`traffic_light_present` 기준으로 crossing classification baseline을 두 그룹으로 나눠 실행했습니다.

출력 파일:

```text
data/features/jaad_features_yolo_bbox_signal_present.csv
data/features/jaad_features_yolo_bbox_signal_absent.csv
outputs/results/baseline_results_crossing_signal_present.csv
outputs/results/baseline_results_crossing_signal_absent.csv
outputs/results/baseline_results_signal_presence_comparison.csv
outputs/results/baseline_results_signal_presence_f1_comparison.csv
```

데이터 크기:

| group | rows | train rows | test rows |
|---|---:|---:|---:|
| signal present | 1,938 | 1,195 | 743 |
| signal absent | 24,677 | 14,198 | 10,479 |

F1 비교:

| feature_set | signal_present f1 | signal_absent f1 |
|---|---:|---:|
| bbox_only | 0.6667 | 0.6607 |
| pose_only | 0.6912 | 0.7190 |
| road_relation_only | 0.6999 | 0.7214 |
| pose_road_relation | 0.6882 | 0.7212 |
| pose_road_signal | 0.6828 | 0.7216 |

해석:

- signal present subset은 1,938 rows로 전체의 일부이며, test video도 3개뿐이라 metric 안정성이 낮습니다.
- 이번 split에서는 pose/road 계열 F1이 signal absent 그룹에서 더 높았습니다.
- signal present 결과는 전체 baseline 대체가 아니라 신호등이 있는 장면에 대한 subset 분석으로 보는 것이 적절합니다.

### YOLO pose 품질 비교

| inference_mode | total detections | success | fallback | success rate |
|---|---:|---:|---:|---:|
| bbox | 26,615 | 19,421 | 7,194 | 0.7297 |
| full_frame | 26,615 | 11,947 | 14,668 | 0.4489 |

### bbox vs full_frame baseline 비교

| feature_set | bbox f1 | full_frame f1 |
|---|---:|---:|
| bbox_only | 0.6560 | 0.6560 |
| pose_only | 0.7108 | 0.7052 |
| road_relation_only | 0.7200 | 0.7198 |
| pose_road_relation | 0.7102 | 0.7013 |

해석:

- bbox crop mode의 YOLO keypoint success rate가 72.97%로, full_frame mode의 44.89%보다 높습니다.
- pose feature를 사용하는 `pose_only`, `pose_road_relation` 성능도 bbox mode가 더 좋습니다.
- 따라서 이후 기본 실험 결과는 bbox crop YOLO pose inference를 기준으로 사용하는 것이 적절합니다.

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
| `main.py` | 실행 entrypoint. mode별 기능 실행, pose inference mode 옵션 지원 |
| `src/jaad_loader.py` | JAAD annotation 및 split 파일 로더 |
| `src/dataset_builder.py` | JAAD feature CSV 생성 파이프라인 |
| `src/pose_extractor.py` | dummy 또는 YOLOv8 pose feature 추출, pose 품질 이벤트 기록 |
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

bbox mode 전체 feature 생성:

```powershell
.\.venv\Scripts\python.exe main.py --mode jaad-features --pose-inference-mode bbox
```

bbox mode baseline 실행:

```powershell
.\.venv\Scripts\python.exe main.py --mode run-baselines --pose-inference-mode bbox
```

full_frame mode 전체 feature 생성:

```powershell
.\.venv\Scripts\python.exe main.py --mode jaad-features --pose-inference-mode full_frame
```

full_frame mode baseline 실행:

```powershell
.\.venv\Scripts\python.exe main.py --mode run-baselines --pose-inference-mode full_frame
```

품질 리포트 생성:

```powershell
.\.venv\Scripts\python.exe main.py --mode quality-report --csv-path data\features\jaad_features_yolo_bbox.csv
```

LSTM 학습:

```powershell
.\.venv\Scripts\python.exe main.py --mode train-lstm --csv-path data\features\jaad_features.csv --feature-set pose_road_relation
```

## 8. 다음에 하면 좋은 작업

1. YOLO pose 결과를 더 안정적으로 만들기
   - bbox crop 방식과 full frame 방식 비교는 완료했습니다.
   - 현재 결과상 bbox crop mode를 기본값으로 사용하는 것이 적절합니다.
   - 다음에는 keypoint confidence threshold 조정과 fallback 발생 frame 분석을 진행하면 좋습니다.

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
   - `baseline_results_crossing.csv`
   - `baseline_results_risk.csv`
   - `lstm_results.csv`
   - `experiment_summary.csv`
   - README Results 표 업데이트

6. risk prediction 확장
   - traffic light가 차량용인지 보행자용인지 구분 불가능한 한계를 명시해야 합니다.
   - 현재 `risk_label`은 traffic context 기반 weak label입니다.
   - 전체 JAAD 기준 risk_label 분포를 확인한 뒤 학습 결과를 해석해야 합니다.

## 9. 현재 주의할 점

- 현재 결과는 YOLOv8n pose 기반이지만 road relation은 아직 완전한 segmentation 기반이 아닙니다.
- traffic light context는 frame-level annotation이며 차량용/보행자용 구분은 제공하지 않습니다.
- `risk_label=-1`은 학습에서 제외되도록 구현했습니다.
- README의 Limitations에 dummy/단순 road relation 한계를 명시했습니다.
- `jaad_features.csv`는 전체 실험의 핵심 입력 파일이지만 Git에는 올리지 않습니다.
- VS Code를 닫고 다시 시작해도 이 문서와 README를 보면 현재 상태를 복원할 수 있습니다.

## 10. SegFormer Cityscapes road segmentation update

- `src/road_segmenter.py`의 `segformer` backend를 HuggingFace pretrained 모델로 실제 동작하도록 연결했습니다.
- 사용 모델은 `nvidia/segformer-b2-finetuned-cityscapes-1024-1024`입니다.
- 모델은 git clone으로 받지 않고 `transformers`의 `SegformerImageProcessor.from_pretrained`와 `SegformerForSemanticSegmentation.from_pretrained`로 로드합니다.
- 최초 실행 시 HuggingFace cache에 자동 다운로드되고, 이후 실행은 cache를 재사용합니다.
- Cityscapes class id 기준으로 `road_class_ids=[0]`, `sidewalk_class_ids=[1]`을 사용합니다.
- 정상 동작 시 feature CSV에는 `segmentation_backend_requested=segformer`, `segmentation_backend=segformer`, `segmentation_source=segformer`, `segmentation_fallback=0`이 기록됩니다.
- transformers 미설치, 모델 다운로드/로드 실패, 추론 실패, 빈 road mask 발생 시 기존 dummy fallback을 사용하고 `fallback_reason`을 road segmentation report에 기록합니다.
- 1프레임 예제 실행 결과: `backend_used=segformer`, `source=segformer`, `fallback=0`, `road_pixel_ratio=0.0113`, preview mask 저장 완료.
- `--limit-videos 5` sanity check 결과: 456 feature rows 모두 `segmentation_backend=segformer`, `segmentation_fallback=0`으로 기록되었습니다.
- 같은 CSV에서 `road_pixel_ratio`는 평균 0.1763, 최소 0.0298, 최대 0.2615로 dummy의 0.4500 고정값이 아닙니다.
- `outputs/reports/road_relation_comparison.csv`에서 dummy와 SegFormer의 road relation feature 분포 차이를 확인했습니다.

검증 명령:

```powershell
cd D:\reserch\jaywalking-risk-recognition

.\.venv\Scripts\python.exe scripts\run_road_segmentation_example.py --backend segformer

.\.venv\Scripts\python.exe main.py --mode jaad-features --pose-inference-mode bbox --segmentation-backend segformer --limit-videos 5 --output-csv data\features\test_jaad_features_road_segformer_5.csv

.\.venv\Scripts\python.exe scripts\compare_road_relation_features.py --dummy-csv data\features\test_jaad_features_road_dummy_5.csv --backend-csv data\features\test_jaad_features_road_segformer_5.csv --backend-name segformer
```

성공 기준:

- `segmentation_backend_requested = segformer`
- `segmentation_backend = segformer`
- `segmentation_source = segformer`
- `segmentation_fallback = 0`
- `road_pixel_ratio`가 dummy의 `0.4500` 고정값이 아니라 frame별 mask 비율로 기록됨
- preview mask가 실제 도로 영역처럼 보임
- `outputs/reports/road_relation_comparison.csv`에서 dummy와 SegFormer의 road relation feature 분포 차이가 확인됨

## 11. NEXT RESUME - SegFormer handoff

작업을 중단한 정확한 지점:

- SegFormer Cityscapes backend 구현은 완료되었습니다.
- `scripts/run_road_segmentation_example.py --backend segformer`는 한 번 실제 모델로 성공했습니다.
- `main.py --mode jaad-features --segmentation-backend segformer --limit-videos 5`도 한 번 실제 모델로 성공했습니다.
- 성공 산출물:
  - `data/features/test_jaad_features_road_segformer_5.csv`
  - `outputs/reports/road_segmentation_report_segformer.csv`
  - `outputs/reports/road_segmentation_summary_segformer.txt`
  - `outputs/reports/road_relation_comparison.csv`
  - `outputs/figures/road_segmentation_example_segformer_mask.png`

성공했던 핵심 수치:

```text
SegFormer 5-video sanity check
feature rows: 456
segmentation_backend_requested: segformer
segmentation_backend: segformer
segmentation_source: segformer
segmentation_fallback: 0
road_pixel_ratio mean/min/max: 0.1763 / 0.0298 / 0.2615

Dummy comparison
dummy road_pixel_ratio: 0.4500 fixed
dummy_vs_segformer road_pixel_ratio mean_abs_diff: 0.2737
```

마지막에 발견한 이슈:

- 현재 가상환경은 `torch 2.5.1+cu121`입니다.
- 최신 `transformers` 계열은 PyTorch 2.6 미만에서 `.bin` weight 로딩을 차단하는 안전 제한이 있습니다.
- 그래서 SegFormer 모델 로딩은 `use_safetensors=True`를 우선 사용하도록 `src/road_segmenter.py`를 맞춰두었습니다.
- HuggingFace cache/network 접근이 sandbox에서 막히면 `from_pretrained`가 cache 확인 요청을 하다가 실패할 수 있습니다.
- 실패하면 기존 설계대로 dummy fallback이 동작하고, `fallback_reason`에 원인이 기록됩니다.

다음에 가장 먼저 할 일:

1. 현재 패키지 버전 확인

```powershell
cd D:\reserch\jaywalking-risk-recognition
.\.venv\Scripts\python.exe -c "import torch, transformers; print(torch.__version__); print(transformers.__version__)"
```

2. SegFormer 1-frame preview 재확인

```powershell
.\.venv\Scripts\python.exe scripts\run_road_segmentation_example.py --backend segformer
```

성공 기준:

```text
backend used: segformer
source: segformer
fallback: 0
road pixel ratio: 0.4500이 아닌 값
mask preview: outputs/figures/road_segmentation_example_segformer_mask.png
```

3. 만약 torch 안전 제한 또는 HuggingFace cache 이슈로 실패하면 둘 중 하나를 선택합니다.

선택 A: PyTorch를 2.6 이상 CUDA build로 업그레이드한 뒤 재실행

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

선택 B: `src/road_segmenter.py`에서 SegFormer 로딩을 safetensors revision/cache 경로로 더 강하게 고정

```text
대상 함수: RoadSegmenter._load_segformer()
핵심: SegformerForSemanticSegmentation.from_pretrained(..., use_safetensors=True)
필요하면 HuggingFace cached snapshot 또는 revision을 명시
```

4. 5-video sanity check 재실행

```powershell
.\.venv\Scripts\python.exe main.py --mode jaad-features --pose-inference-mode bbox --segmentation-backend segformer --limit-videos 5 --output-csv data\features\test_jaad_features_road_segformer_5.csv
```

5. CSV metadata 확인

```powershell
Import-Csv data\features\test_jaad_features_road_segformer_5.csv |
  Group-Object segmentation_backend,segmentation_backend_requested,segmentation_source,segmentation_fallback |
  Select-Object Count,Name

Import-Csv data\features\test_jaad_features_road_segformer_5.csv |
  Measure-Object road_pixel_ratio -Average -Minimum -Maximum
```

6. dummy와 분포 비교

```powershell
.\.venv\Scripts\python.exe scripts\compare_road_relation_features.py --dummy-csv data\features\test_jaad_features_road_dummy_5.csv --backend-csv data\features\test_jaad_features_road_segformer_5.csv --backend-name segformer
```

다음 작업의 완료 기준:

- `road_segmentation_example_segformer.txt`에서 `fallback: 0`
- `test_jaad_features_road_segformer_5.csv` 전체 row에서 `segmentation_backend=segformer`
- `segmentation_fallback` 합계가 0
- `road_pixel_ratio`가 0.4500 고정이 아님
- `road_relation_comparison.csv`에서 dummy와 SegFormer feature 차이 확인

주의:

- SegFormer가 실패해도 pipeline 자체는 깨지지 않고 dummy fallback으로 CSV가 만들어집니다.
- 따라서 다음에는 반드시 CSV metadata와 `road_segmentation_summary_segformer.txt`를 확인해야 합니다.
- `segmentation_backend_requested=segformer`만 보고 성공으로 판단하면 안 됩니다. 실제 성공은 `segmentation_backend=segformer`, `segmentation_fallback=0`입니다.

## 12. 2026-05-04 SegFormer 재실행 업데이트

오늘 멈췄던 SegFormer road segmentation sanity check를 다시 실행했습니다.

### 실행 환경 확인

```text
torch: 2.5.1+cu121
transformers: 4.56.2
CUDA available: True
GPU: NVIDIA GeForce GTX 1650 SUPER
YOLO weights: models/yolo/yolov8n-pose.pt exists
dummy comparison CSV: data/features/test_jaad_features_road_dummy_5.csv exists
```

### 1-frame SegFormer preview 재실행

명령:

```powershell
.\.venv\Scripts\python.exe scripts\run_road_segmentation_example.py --backend segformer
```

처음 sandbox 안에서 실행했을 때는 HuggingFace 접근 제한 때문에 dummy fallback으로 떨어졌습니다. 네트워크 접근을 허용한 뒤 재실행했을 때는 SegFormer 모델이 정상 로드되었습니다.

성공 결과:

```text
backend requested: segformer
backend used: segformer
source: segformer
fallback: 0
fallback reason: none
road pixel ratio: 0.0113
mask preview: outputs/figures/road_segmentation_example_segformer_mask.png
```

출력 파일:

```text
outputs/reports/road_segmentation_example_segformer.csv
outputs/reports/road_segmentation_example_segformer.txt
outputs/figures/road_segmentation_example_segformer_mask.png
```

### 5-video SegFormer sanity check 재실행

명령:

```powershell
.\.venv\Scripts\python.exe main.py --mode jaad-features --pose-inference-mode bbox --segmentation-backend segformer --limit-videos 5 --output-csv data\features\test_jaad_features_road_segformer_5.csv
```

이 명령도 sandbox 안에서는 HuggingFace 접근 제한 때문에 dummy fallback으로 실행되었습니다. 네트워크 접근을 허용해 다시 실행했고, 실제 SegFormer backend로 feature CSV를 재생성했습니다.

성공 결과:

```text
JAAD videos: 5/5
failed videos: 0
feature rows: 456
split rows: train 187, val 69, test 200
pose detections: 456
YOLO pose success: 395
YOLO pose fallback: 61
YOLO pose success rate: 0.8662
```

Segmentation summary:

```text
backend requested: segformer
backend active: segformer
load error: none
total frames: 253
model frames: 253
dummy frames: 0
fallback frames: 0
fallback ratio: 0.0000
average road pixel ratio: 0.1980
average sidewalk pixel ratio: 0.0156
backend used counts: {'segformer': 253}
```

Feature CSV metadata 확인:

```text
456 rows: segmentation_backend=segformer,
          segmentation_backend_requested=segformer,
          segmentation_source=segformer,
          segmentation_fallback=0

road_pixel_ratio count: 456
road_pixel_ratio average: 0.17630658013185
road_pixel_ratio minimum: 0.0297545331790123
road_pixel_ratio maximum: 0.261534529320988
```

출력 파일:

```text
data/features/test_jaad_features_road_segformer_5.csv
outputs/reports/road_segmentation_report_segformer.csv
outputs/reports/road_segmentation_summary_segformer.txt
outputs/reports/pose_detection_report_yolo_bbox.csv
outputs/reports/pose_detection_summary_yolo_bbox.txt
outputs/reports/jaad_data_quality_report.csv
outputs/reports/jaad_data_quality_summary.txt
```

### dummy vs SegFormer road relation 비교 갱신

명령:

```powershell
.\.venv\Scripts\python.exe scripts\compare_road_relation_features.py --dummy-csv data\features\test_jaad_features_road_dummy_5.csv --backend-csv data\features\test_jaad_features_road_segformer_5.csv --backend-name segformer
```

출력:

```text
outputs/reports/road_relation_comparison.csv
```

핵심 비교:

```text
dummy road_pixel_ratio mean: 0.4500 fixed
segformer road_pixel_ratio mean/min/max: 0.1763 / 0.0298 / 0.2615
dummy_vs_segformer road_pixel_ratio mean_abs_diff: 0.2737

dummy distance_to_road mean: 0.0
segformer distance_to_road mean: 54.5193

dummy foot_on_road mean: 1.0
segformer foot_on_road mean: 0.3026

dummy center_on_road mean: 1.0
segformer center_on_road mean: 0.0066
```

해석:

- SegFormer sanity check는 실제 backend 기준으로 성공했습니다.
- 이번 결과는 `segmentation_backend_requested=segformer`만 찍힌 것이 아니라, 전체 row에서 `segmentation_backend=segformer`, `segmentation_fallback=0`으로 확인되었습니다.
- dummy road mask는 5-video 샘플에서 `road_pixel_ratio=0.4500`, `distance_to_road=0`, `foot_on_road=1`, `center_on_road=1`처럼 거의 고정적인 feature를 만들었습니다.
- SegFormer는 frame별 road mask를 반영해 road relation feature 분포가 실제로 달라졌습니다.
- 따라서 다음 핵심 작업은 전체 JAAD 346개 영상에 SegFormer backend를 적용한 feature CSV를 생성하고, 그 CSV로 RandomForest baseline을 다시 실행하는 것입니다.

### 다음 작업

전체 JAAD SegFormer feature 생성:

```powershell
.\.venv\Scripts\python.exe main.py --mode jaad-features --pose-inference-mode bbox --segmentation-backend segformer --output-csv data\features\jaad_features_yolo_bbox_segformer.csv
```

완료 후 반드시 확인할 것:

```powershell
Import-Csv data\features\jaad_features_yolo_bbox_segformer.csv |
  Group-Object segmentation_backend,segmentation_backend_requested,segmentation_source,segmentation_fallback |
  Select-Object Count,Name

Import-Csv data\features\jaad_features_yolo_bbox_segformer.csv |
  Measure-Object road_pixel_ratio -Average -Minimum -Maximum
```

그다음 SegFormer feature CSV 기준 baseline 실행:

```powershell
.\.venv\Scripts\python.exe main.py --mode run-baselines --csv-path data\features\jaad_features_yolo_bbox_segformer.csv --target-column label --baseline-output outputs\results\baseline_results_crossing_segformer.csv
```

주의:

- SegFormer 전체 실행도 sandbox/network 제한이 있으면 dummy fallback으로 떨어질 수 있습니다.
- 전체 실행 후에도 반드시 `segmentation_backend=segformer`, `segmentation_fallback=0`을 확인해야 합니다.
- 전체 결과가 나오면 `pose_only` vs `pose_road_relation`을 dummy 기반 결과와 SegFormer 기반 결과로 비교하는 것이 연구 질문에 가장 직접적인 답이 됩니다.

## 13. 2026-05-04 전체 JAAD SegFormer baseline 결과

전체 JAAD 346개 영상에 대해 bbox crop YOLO pose와 SegFormer road segmentation backend를 사용한 feature 생성을 완료했습니다.

실행 명령:

```powershell
.\.venv\Scripts\python.exe main.py --mode jaad-features --pose-inference-mode bbox --segmentation-backend segformer --output-csv data\features\jaad_features_yolo_bbox_segformer.csv
```

Segmentation metadata 확인:

```text
26607 rows: segmentation_backend=segformer,
            segmentation_backend_requested=segformer,
            segmentation_source=segformer,
            segmentation_fallback=0

8 rows: segmentation_backend=dummy,
        segmentation_backend_requested=segformer,
        segmentation_source=dummy,
        segmentation_fallback=1
```

해석:

- 전체 26,615 feature rows 중 26,607 rows가 실제 SegFormer backend로 생성되었습니다.
- dummy fallback은 8 rows뿐이며, 전체 row 기준 약 0.03%입니다.
- 따라서 전체 feature CSV는 사실상 SegFormer road mask 기반 결과로 해석할 수 있습니다.

Road pixel ratio:

```text
count: 26,615
average: 0.23291214419634
minimum: 0.0000260416666666667
maximum: 0.45208912037037
```

기존 dummy backend는 `road_pixel_ratio`가 거의 0.4500 고정이었지만, SegFormer 전체 결과는 frame별 road mask 비율을 반영했습니다.

SegFormer CSV 기준 RandomForest crossing baseline:

실행 명령:

```powershell
.\.venv\Scripts\python.exe main.py --mode run-baselines --csv-path data\features\jaad_features_yolo_bbox_segformer.csv --target-column label --baseline-output outputs\results\baseline_results_crossing_segformer.csv
```

결과:

| feature_set | model | accuracy | precision | recall | f1 | train_rows | test_rows |
|---|---|---:|---:|---:|---:|---:|---:|
| bbox_only | RandomForest | 0.6043 | 0.6395 | 0.6733 | 0.6560 | 15,393 | 11,222 |
| pose_only | RandomForest | 0.6631 | 0.6847 | 0.7389 | 0.7108 | 15,393 | 11,222 |
| road_relation_only | RandomForest | 0.6260 | 0.6625 | 0.6778 | 0.6701 | 15,393 | 11,222 |
| pose_road_relation | RandomForest | 0.7217 | 0.7460 | 0.7632 | 0.7545 | 15,393 | 11,222 |
| pose_road_signal | RandomForest | 0.7281 | 0.7571 | 0.7580 | 0.7575 | 15,393 | 11,222 |

기존 dummy 기반 bbox YOLO baseline과 비교:

| feature_set | dummy F1 | SegFormer F1 | 변화 |
|---|---:|---:|---:|
| bbox_only | 0.6560 | 0.6560 | +0.0000 |
| pose_only | 0.7108 | 0.7108 | +0.0000 |
| road_relation_only | 0.7200 | 0.6701 | -0.0499 |
| pose_road_relation | 0.7102 | 0.7545 | +0.0443 |
| pose_road_signal | 0.7217 | 0.7575 | +0.0358 |

중요 해석:

- `bbox_only`와 `pose_only`는 road mask를 사용하지 않으므로 기존 결과와 동일합니다.
- dummy backend에서는 `pose_only` F1 0.7108과 `pose_road_relation` F1 0.7102가 거의 같아 road relation의 이점이 드러나지 않았습니다.
- SegFormer backend에서는 `pose_road_relation` F1이 0.7545로 상승해 `pose_only`보다 0.0437 높아졌습니다.
- 따라서 연구 질문인 "pose feature에 road relation feature를 결합하면 더 유용한가?"에 대해, 실제 road segmentation을 적용했을 때는 "그렇다"고 말할 수 있는 근거가 생겼습니다.
- `road_relation_only`는 dummy에서 F1이 높아 보였지만 recall 편향이 강했습니다. SegFormer에서는 F1이 낮아졌지만 precision/recall이 더 균형적으로 바뀌었습니다.
- `pose_road_signal`은 SegFormer 기준 F1 0.7575로 가장 높지만, traffic signal context는 차량용/보행자용 신호 구분 한계가 있으므로 보조 feature로 해석하는 것이 안전합니다.

문서 반영 상태:

- `README.md` Results와 Discussion에 SegFormer 전체 baseline 결과를 반영했습니다.
- `docs/paper_method_evidence.md`에 논문용 SegFormer 결과 해석과 문장 초안을 추가했습니다.

다음 작업:

1. SegFormer feature CSV로 LSTM 재학습 여부 결정
2. 이후 ST-GCN 또는 pose graph 모델에서 SegFormer road relation feature를 사용할지 결정

## 14. 2026-05-04 RandomForest diagnostics export

RandomForest 진단 산출물을 한 번에 생성하는 스크립트를 추가했습니다.

추가 파일:

```text
scripts/export_rf_diagnostics.py
```

실행 명령:

```powershell
.\.venv\Scripts\python.exe scripts\export_rf_diagnostics.py
```

기본 입력:

```text
feature CSV: data/features/jaad_features_yolo_bbox_segformer.csv
experiment baseline: outputs/results/baseline_results_crossing_segformer.csv
comparison baseline: outputs/results/baseline_results_crossing.csv
target column: label
experiment name: segformer
comparison name: dummy
```

생성 산출물:

```text
outputs/reports/rf_diagnostics/confusion_matrices_segformer_label.csv
outputs/reports/rf_diagnostics/confusion_matrix_summary_segformer_label.csv
outputs/reports/rf_diagnostics/feature_importance_segformer_label.csv
outputs/reports/rf_diagnostics/feature_group_importance_segformer_label.csv
outputs/reports/rf_diagnostics/ablation_dummy_vs_segformer_label.csv
outputs/figures/rf_diagnostics/random_forest_segformer_*_label_confusion_matrix.png
```

핵심 진단 결과:

```text
pose_only confusion:
  TN 2795, FP 2139, FN 1642, TP 4646

pose_road_relation confusion:
  TN 3300, FP 1634, FN 1489, TP 4799

pose_road_signal confusion:
  TN 3405, FP 1529, FN 1522, TP 4766
```

해석:

- SegFormer `pose_road_relation`은 `pose_only`보다 false positive를 505개 줄였습니다.
- SegFormer `pose_road_relation`은 `pose_only`보다 false negative를 153개 줄였습니다.
- 즉 F1 상승은 한쪽 오류만 줄인 결과가 아니라 FP와 FN이 함께 줄어든 결과입니다.

`pose_road_relation` feature importance 상위 feature:

| rank | feature | group | importance |
|---:|---|---|---:|
| 1 | distance_to_road | road_relation | 0.1528 |
| 2 | center_y | bbox_position | 0.1227 |
| 3 | left_ankle_y | pose | 0.0981 |
| 4 | right_ankle_y | pose | 0.0963 |
| 5 | left_ankle_x | pose | 0.0909 |

`pose_road_relation` grouped importance:

| group | importance |
|---|---:|
| pose | 0.5308 |
| road_relation | 0.2600 |
| bbox_position | 0.2092 |

논문/보고서용 해석:

- `distance_to_road`가 가장 중요한 개별 feature로 나왔습니다.
- road relation feature group이 전체 중요도의 약 26.0%를 차지했습니다.
- 이는 SegFormer 적용 후 `pose_road_relation` 성능 향상이 실제 road geometry feature 활용과 연결되어 있음을 뒷받침합니다.
