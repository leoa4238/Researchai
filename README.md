# Jaywalking Risk Recognition

JAAD(Joint Attention in Autonomous Driving) 데이터셋을 이용해 보행자의 무단횡단 위험 행동을 인식하는 연구용 Python 파이프라인입니다. 본 프로젝트는 보행자의 위치와 자세 정보뿐 아니라, 보행자와 도로 영역 사이의 공간 관계를 함께 사용했을 때 crossing 행동 인식 성능이 어떻게 달라지는지 실험하는 것을 목표로 합니다.

현재 구현은 JAAD annotation의 보행자 bounding box와 crossing label을 기반으로 feature CSV를 생성하고, JAAD official split을 사용해 RandomForest 및 LSTM baseline을 학습/평가합니다.

## Research Question

본 연구의 핵심 질문은 다음과 같습니다.

> 보행자의 pose 정보에 road relation feature를 결합하면, pose-only feature만 사용할 때보다 무단횡단 위험 행동 인식에 더 유용한가?

무단횡단 위험은 단순히 사람이 걷고 있는지, 어느 방향을 보고 있는지로만 결정되지 않습니다. 같은 walking pose라도 보행자가 인도 위에 있는지, 도로에 진입했는지, 도로와의 거리가 줄어드는지에 따라 위험도는 달라집니다. 따라서 본 연구는 다음 두 가지 정보를 함께 고려합니다.

- **Pose feature**: 보행자의 중심점, 발목 위치, 신체 방향, 보행 방향
- **Road relation feature**: 도로까지의 거리, 발/중심점의 도로 진입 여부, 도로 접근 속도

이 조합을 통해 보행자의 행동 자체와 도로 환경 내 위치 관계를 함께 반영하는 위험 행동 인식 모델을 구성합니다.

## Contributions

본 프로젝트의 현재 기여는 다음과 같습니다.

1. **JAAD annotation 기반 feature generation pipeline 구축**  
   JAAD XML annotation을 직접 파싱하여 frame별 pedestrian bbox, crossing label, action, look, occlusion 정보를 feature CSV로 변환합니다.

2. **Official split 기반 실험 환경 구성**  
   JAAD `split_ids`의 train/val/test split을 읽어 `jaad_features.csv`에 `split` 컬럼을 추가하고, 학습/평가 시 video-level leakage가 발생하지 않도록 검증합니다.

3. **Pose와 road relation feature 비교 실험 지원**  
   `bbox_only`, `pose_only`, `road_relation_only`, `pose_road_relation` feature set을 정의하고 RandomForest baseline을 자동 실행할 수 있습니다.

4. **데이터 품질 검증 및 리포트 생성 기능 추가**  
   전체 row 수, split 분포, label 분포, 결측치, feature 통계, video overlap 여부를 분석해 CSV 및 텍스트 리포트로 저장합니다.

## Experimental Setup

### Dataset

본 연구는 JAAD 2.0 데이터셋을 사용합니다.

- Dataset: JAAD
- Video clips: 346개
- Annotation format: XML
- Label source: pedestrian `cross` attribute
- Positive label: `crossing`
- Negative label: `not-crossing`

현재 데이터 위치:

```text
data/raw/JAAD/
  JAAD_clips/
    video_0001.mp4
    video_0002.mp4
    ...

data/raw/JAAD_annotations/
  annotations/
  annotations_appearance/
  annotations_attributes/
  annotations_traffic/
  annotations_vehicle/
  split_ids/
  jaad_data.py
```

### Split

기본 실험은 JAAD official split을 사용합니다.

현재 기본 설정은 전체 346개 영상을 포함하는 `all_videos` split입니다.

```yaml
jaad:
  split_subset: all_videos

training:
  split_strategy: official
```

`all_videos` split 구성:

| Split | Video Count |
|---|---:|
| train | 188 |
| val | 32 |
| test | 126 |

학습 시 `train + val`을 학습 데이터로 사용하고, `test`를 평가 데이터로 사용합니다. 학습 전 `video_id`가 train/val/test 사이에 중복으로 들어가지 않는지 검증합니다.

### Feature Sets

현재 지원하는 feature set은 다음과 같습니다.

| Feature Set | Included Features |
|---|---|
| `bbox_only` | `center_x`, `center_y` |
| `pose_only` | bbox 중심, 양쪽 발목 좌표, body direction, step direction |
| `road_relation_only` | `distance_to_road`, `foot_on_road`, `center_on_road`, `approach_rate` |
| `pose_road_relation` | pose feature + road relation feature |

## Method

JAAD feature 생성 파이프라인은 다음 순서로 동작합니다.

```text
JAAD XML annotation
  -> video_id별 annotation 로드
  -> frame별 pedestrian bbox 추출
  -> cross 속성을 label로 변환
  -> official split 파일 기반 train/val/test 부여
  -> bbox를 Detection 객체로 변환
  -> dummy pose 또는 YOLO pose 추출
  -> road/sidewalk segmentation
  -> pose + road relation feature 계산
  -> data/features/jaad_features.csv 저장
  -> 품질 리포트 생성
  -> baseline 모델 학습 및 평가
```

현재 JAAD 모드는 로컬 `yolov8n-pose.pt` weights를 사용해 YOLO pose keypoint를 추출합니다. road/sidewalk segmentation은 아직 dummy road mask를 사용하며, 이후 실제 road/sidewalk segmentation 모델로 교체할 수 있도록 구조를 분리해두었습니다.

YOLO pose를 사용할 때는 weights를 인터넷에서 자동 다운로드하지 않습니다. `yolov8n-pose.pt` 또는 `yolov8s-pose.pt` 파일을 직접 받아서 `models/yolo/` 아래에 두고 `configs/default.yaml`의 `pose.model_path`를 해당 파일로 지정해야 합니다. weights 파일이 없으면 로그를 출력하고 dummy pose fallback을 사용합니다.

## Results

아래 결과는 `--limit-videos` 없이 전체 JAAD 346개 영상을 대상으로 YOLOv8 pose 기반 feature 생성을 수행한 뒤 얻은 실험 결과입니다.

### Data Quality Summary

전체 영상 346개를 처리 대상으로 실행했으며, 실패한 영상은 없었습니다. 현재 `sample_type: beh` 설정으로 behavior annotation이 있는 pedestrian만 feature row로 생성되므로, 최종 feature row가 생성된 video 수는 320개입니다.

| Metric | Value |
|---|---:|
| Attempted videos | 346 |
| Failed videos | 0 |
| Videos with feature rows | 320 |
| Total rows | 26,615 |
| Pedestrians | 686 |
| Missing values | 0 |
| Train rows | 13,269 |
| Val rows | 2,124 |
| Test rows | 11,222 |
| Train videos | 172 |
| Val videos | 30 |
| Test videos | 118 |
| Label 0 | 11,626 (43.68%) |
| Label 1 | 14,989 (56.32%) |
| Train/Test video overlap | 0 |
| Train/Val video overlap | 0 |
| Val/Test video overlap | 0 |

### Baseline Results

아래 표는 `outputs/results/baseline_results.csv` 및 `outputs/results/lstm_results.csv`를 기준으로 작성한 전체 JAAD 실험 결과입니다.

| Feature Set | Model | Accuracy | Precision | Recall | F1 | Train Rows | Test Rows |
|---|---|---:|---:|---:|---:|---:|---:|
| `bbox_only` | RandomForest | 0.6043 | 0.6395 | 0.6733 | 0.6560 | 15,393 | 11,222 |
| `pose_only` | RandomForest | 0.6631 | 0.6847 | 0.7389 | 0.7108 | 15,393 | 11,222 |
| `road_relation_only` | RandomForest | 0.5653 | 0.5633 | 0.9975 | 0.7200 | 15,393 | 11,222 |
| `pose_road_relation` | RandomForest | 0.6626 | 0.6847 | 0.7376 | 0.7102 | 15,393 | 11,222 |
| `pose_road_relation` | LSTM | 0.6601 | 0.7647 | 0.6404 | 0.6971 | 12,659 seq. | 9,184 seq. |

### LSTM Sequence Statistics

| Split | Sequences | Label 0 | Label 1 | Excluded Short Tracks |
|---|---:|---:|---:|---:|
| Train | 12,659 | 4,917 | 7,742 | 12 |
| Test | 9,184 | 3,575 | 5,609 | 5 |

## Discussion

전체 JAAD 실험 결과에서 `pose_only` RandomForest는 `bbox_only`보다 높은 F1을 보였습니다. 이는 보행자의 중심 위치만 사용하는 것보다 YOLO pose에서 얻은 발목 위치, body direction, step direction과 같은 pose-derived feature가 crossing classification에 추가 정보를 제공한다는 점을 시사합니다.

`road_relation_only`는 accuracy는 낮지만 recall이 매우 높게 나타났습니다. 이는 현재 dummy road mask가 도로 접근 또는 도로 위 여부를 넓게 포착하면서 positive class를 적극적으로 예측하는 경향을 만든 것으로 해석할 수 있습니다. 이 경우 F1은 높지만 precision이 낮으므로, 실제 위험 탐지에서 false positive를 줄이려면 road segmentation 품질 개선이 필요합니다.

RandomForest 기준 `pose_only`와 `pose_road_relation`의 F1은 거의 비슷했습니다. 현재 road relation feature가 dummy road segmentation에 기반하므로, road relation의 잠재적 이점이 충분히 드러나지 않았을 가능성이 큽니다. 이번 LSTM `pose_road_relation` 결과는 F1 0.6971로 RandomForest pose 계열보다 약간 낮았습니다. 다만 precision은 0.7647로 가장 높아, sequence model이 더 보수적으로 positive class를 예측하는 경향을 보였습니다.

## Limitations

현재 구현에는 다음 한계가 있습니다.

- **YOLO pose 품질 의존성**: 현재 JAAD 모드는 로컬 YOLOv8n pose weights를 사용합니다. bbox crop 기반 inference에서 keypoint가 검출되지 않으면 dummy pose fallback을 사용하므로, pose feature 품질은 YOLO 검출 품질에 영향을 받습니다.
- **Dummy road segmentation**: 현재 road mask는 화면 하단 영역을 도로로 가정하는 단순 heuristic입니다. 실제 도로/인도 경계를 정확히 반영하지 못합니다.
- **Behavior-only row 생성**: 전체 346개 영상을 처리했지만 `sample_type: beh` 설정 때문에 feature row가 생성된 video는 320개입니다.
- **Label 정의의 단순성**: 현재 label은 JAAD `cross` 속성만 사용합니다. action, look, traffic, vehicle annotation은 아직 위험도 정의에 통합되지 않았습니다.
- **Temporal modeling 제한**: LSTM baseline은 구현되어 있으나 sequence 구성, normalization, imbalance 처리, split 전략을 더 정교화할 필요가 있습니다.

## Future Work

다음 단계에서는 아래 개선을 목표로 합니다.

1. **YOLO pose 적용**  
   `yolov8n-pose.pt` 또는 더 강한 pose 모델을 로컬에 준비하고 `jaad.pose_backend: yolo`로 전환합니다.

2. **실제 road/sidewalk segmentation 적용**  
   dummy road mask를 segmentation model 또는 annotation 기반 road mask로 교체해 `distance_to_road`, `foot_on_road`, `center_on_road` feature의 신뢰도를 높입니다.

3. **Full-result table 자동 갱신**  
   `baseline_results.csv`, `lstm_results.csv`, `experiment_summary.csv`를 README Markdown table로 자동 반영하는 스크립트를 추가합니다.

4. **ST-GCN 기반 시공간 pose 모델 검토**  
   보행자의 keypoint sequence를 graph 구조로 표현해 ST-GCN 또는 유사한 spatio-temporal graph model을 적용합니다.

5. **Context annotation 통합**  
   traffic light, vehicle action, pedestrian look/action 정보를 결합해 crossing 여부뿐 아니라 위험 상황 예측으로 확장합니다.

## Installation

```powershell
cd D:\reserch\jaywalking-risk-recognition
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

이미 `.venv`가 있다면 다음처럼 실행할 수 있습니다.

```powershell
cd D:\reserch\jaywalking-risk-recognition
.\.venv\Scripts\python.exe main.py --help
```

## GPU / CUDA Environment

전체 JAAD feature 생성에서 YOLO pose inference를 사용할 경우 CPU보다 GPU 실행이 훨씬 빠릅니다. 현재 환경은 `.venv` 안에 CUDA 지원 PyTorch를 설치해 GPU 사용이 가능하도록 구성합니다.

GPU 인식 여부 확인:

```powershell
cd D:\reserch\jaywalking-risk-recognition
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA not available')"
```

`torch.cuda.is_available()`가 `True`이고 GPU 이름이 출력되면 GPU 사용 준비가 된 상태입니다.

CUDA 지원 PyTorch 재설치가 필요한 경우:

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y torch torchvision torchaudio
.\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

GPU 사용 대상:

- YOLOv8 pose inference
- LSTM 학습

RandomForest는 scikit-learn 기반이므로 기본적으로 CPU에서 실행됩니다.

## Local YOLO Pose Weights

YOLOv8 pose backend는 로컬 weights 파일만 사용합니다. 자동 다운로드는 수행하지 않습니다.

권장 위치:

```text
models/yolo/yolov8n-pose.pt
models/yolo/yolov8s-pose.pt
```

설정 예:

```yaml
pose:
  backend: yolo
  model_path: models/yolo/yolov8n-pose.pt
  confidence_threshold: 0.25
  inference_mode: bbox

jaad:
  pose_backend: yolo
```

`pose.inference_mode`는 두 가지를 지원합니다.

- `bbox`: annotation 또는 detector bbox 영역을 crop한 뒤 pose keypoint 추출
- `full_frame`: full frame에서 pose를 추출한 뒤 bbox와 매칭

weights 파일이 없거나 keypoint가 검출되지 않으면 bbox 기반 dummy pose로 fallback합니다.

## Usage

### JAAD Feature 생성

전체 346개 영상 처리:

```powershell
.\.venv\Scripts\python.exe main.py --mode jaad-features
```

일부 영상만 처리:

```powershell
.\.venv\Scripts\python.exe main.py --mode jaad-features --limit-videos 20
```

특정 영상 하나만 처리:

```powershell
.\.venv\Scripts\python.exe main.py --mode jaad-features --jaad-video-id video_0002 --limit-videos 1
```

실패한 영상은 다음 파일에 기록됩니다.

```text
outputs/logs/jaad_failed_videos.txt
```

### Data Quality Report 생성

feature CSV 생성 후 자동으로 생성되며, 수동으로 다시 만들 수도 있습니다.

```powershell
.\.venv\Scripts\python.exe main.py --mode quality-report --csv-path data\features\jaad_features.csv
```

출력 파일:

```text
outputs/reports/jaad_data_quality_report.csv
outputs/reports/jaad_data_quality_summary.txt
```

### RandomForest 학습

```powershell
.\.venv\Scripts\python.exe main.py --mode train-rf --csv-path data\features\jaad_features.csv --feature-set pose_road_relation
```

### LSTM 학습

```powershell
.\.venv\Scripts\python.exe main.py --mode train-lstm --csv-path data\features\jaad_features.csv --feature-set pose_road_relation
```

### Baseline 실험 자동 실행

```powershell
.\.venv\Scripts\python.exe main.py --mode run-baselines --csv-path data\features\jaad_features.csv
```

출력 파일:

```text
outputs/results/baseline_results.csv
```

## Data Validation

학습 전 다음 항목을 자동으로 검사합니다.

- label이 한쪽 클래스만 있는지 확인
- train/test에 동일 `video_id`가 섞였는지 확인
- NaN 또는 inf 값이 있는지 확인
- feature column 누락 여부 확인

심각한 문제가 있으면 학습을 중단합니다. test split에 한쪽 label만 있는 경우처럼 metric 해석에 주의가 필요한 상황은 warning으로 출력합니다.

## Project Structure

```text
main.py
  CLI entrypoint

src/jaad_loader.py
  JAAD XML annotation 및 official split loader

src/dataset_builder.py
  feature CSV 생성

src/data_quality.py
  품질 리포트 및 학습 전 검증

src/baseline_runner.py
  feature set별 RandomForest baseline 실행

src/train_random_forest.py
  RandomForest 학습 및 평가

src/train_lstm.py
  LSTM 학습 및 평가

configs/default.yaml
  경로, split, feature set, 학습 설정
```

## References

JAAD 2.0 repository:

```text
https://github.com/ykotseruba/JAAD
```

JAAD video clips:

```text
http://data.nvision2.eecs.yorku.ca/JAAD_dataset/data/JAAD_clips.zip
```
