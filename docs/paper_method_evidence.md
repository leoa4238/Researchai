# Paper Method Evidence

이 문서는 논문/보고서에 넣을 때 사용할 수 있도록, 본 프로젝트가 어떤 output을 근거로 분석했고 어떤 방식으로 feature를 확장했는지 정리한 자료입니다.

## 1. 현재 분석 방식 요약

본 프로젝트는 원본 영상을 end-to-end로 직접 학습하지 않습니다. 대신 JAAD video frame과 annotation을 이용해 보행자 단위 feature row를 만들고, 이 숫자 feature CSV를 RandomForest 또는 LSTM에 입력합니다.

```text
JAAD video frame
-> JAAD pedestrian bbox annotation
-> YOLO pose keypoint extraction
-> road relation feature
-> traffic light context feature
-> feature CSV
-> crossing classification / risk prediction baseline
```

즉 논문에서 보여줘야 할 핵심은 다음입니다.

- 영상이 어떤 숫자 feature로 변환되는지
- 어떤 annotation을 근거로 feature를 추가했는지
- 어떤 CSV와 결과 파일을 근거로 성능을 비교했는지
- 새로 만든 risk_label이 어떤 한계를 갖는지

## 2. 논문에 넣기 좋은 그림

README에 이미 삽입한 아래 그림들을 Method/Experiment 설명에 사용할 수 있습니다.

| 그림 파일 | 논문에서 보여주는 내용 |
|---|---|
| `docs/images/analysis_process_pipeline.png` | 영상 frame이 feature row와 model prediction으로 바뀌는 전체 절차 |
| `docs/images/feature_row_process_capture.png` | 모델이 실제로 보는 feature row 구성 |
| `docs/images/traffic_annotation_analysis_capture.png` | JAAD traffic annotation 구조와 traffic_light 값 분포 |
| `docs/images/pose_quality_capture.png` | YOLO pose extraction 성공/fallback 품질 |
| `docs/images/baseline_results_capture.png` | crossing/risk baseline 결과 요약 |

## 3. Output에서 확인할 부분

### 3.1 Traffic annotation 구조 분석

확인 파일:

```text
outputs/reports/traffic_annotation_analysis.txt
outputs/reports/traffic_annotation_values.csv
```

논문에 쓸 수 있는 핵심 결과:

```text
traffic_light values:
- n/a: 77,599
- green: 2,263
- red: 2,170
```

해석:

- JAAD traffic annotation은 frame별 `traffic_light` 값을 제공합니다.
- red/green 상태는 구분 가능합니다.
- yellow는 현재 annotation에서 관측되지 않았습니다.
- 차량용 신호인지 보행자용 신호인지는 구분되지 않습니다.

### 3.2 Feature CSV 확장 확인

확인 파일:

```text
data/features/jaad_features_yolo_bbox.csv
```

추가된 컬럼:

```text
traffic_light_present
traffic_light_state
traffic_light_state_code
risk_label
```

의미:

| 컬럼 | 의미 |
|---|---|
| `traffic_light_present` | frame에 red/green/yellow 신호 상태가 있으면 1, 없으면 0 |
| `traffic_light_state` | `red`, `green`, `yellow`, `unknown` |
| `traffic_light_state_code` | `unknown=0`, `red=1`, `yellow=2`, `green=3` |
| `risk_label` | traffic signal을 이용해 조건부로 만든 weak risk label |

### 3.3 Data quality 확인

확인 파일:

```text
outputs/reports/jaad_data_quality_summary.txt
outputs/reports/jaad_data_quality_report.csv
```

논문에 쓸 수 있는 핵심 결과:

```text
total rows: 26,615
processed videos: 320
traffic_light_present 0: 24,677
traffic_light_present 1: 1,938
traffic_light_state green: 904
traffic_light_state red: 1,034
traffic_light_state unknown: 24,677
risk_label -1: 25,145
risk_label 0: 904
risk_label 1: 566
risk_label trainable rows: 1,470
```

해석:

- 전체 feature row 중 대부분은 traffic signal 상태가 unknown입니다.
- risk prediction은 `risk_label=-1`을 제외한 1,470 rows만 사용합니다.
- crossing classification은 전체 26,615 rows를 사용합니다.

### 3.4 Road segmentation 확인

확인 파일:

```text
outputs/reports/road_segmentation_report.csv
outputs/reports/road_segmentation_summary.txt
```

feature CSV에서 확인할 컬럼:

```text
segmentation_backend
segmentation_backend_requested
segmentation_source
segmentation_fallback
road_pixel_ratio
```

해석:

- `segmentation_backend_requested`는 사용자가 요청한 backend입니다.
- `segmentation_backend`는 실제 사용된 backend입니다.
- model이 없거나 실패하면 `segmentation_backend=dummy`, `segmentation_fallback=1`로 기록됩니다.
- 현재 기본 설정은 backward compatibility를 위해 dummy road mask입니다.
- 실제 road/sidewalk segmentation checkpoint를 연결하면 기존 road relation feature 계산은 동일하게 road mask를 사용합니다.
- 현재 `models/segmentation/`에는 checkpoint가 없어서 yolo_seg는 dummy fallback으로 검증되었습니다.
- `--limit-videos 5` 기준 yolo_seg 요청 결과는 fallback ratio 1.0000, average road pixel ratio 0.4500으로 기록되었습니다.
- 이후 `segformer` backend는 HuggingFace `nvidia/segformer-b2-finetuned-cityscapes-1024-1024` pretrained Cityscapes 모델을 `from_pretrained` 방식으로 로드하도록 구현되었습니다.
- SegFormer sanity check에서는 `segmentation_backend_requested=segformer`, `segmentation_backend=segformer`, `segmentation_source=segformer`, `segmentation_fallback=0`으로 기록되어 실제 road/sidewalk segmentation mask가 사용되었습니다.
- `--limit-videos 5` 기준 SegFormer feature CSV의 `road_pixel_ratio`는 평균 0.1763, 최소 0.0298, 최대 0.2615로 dummy의 0.4500 고정값과 다르게 frame별 mask 비율을 반영했습니다.
- 전체 JAAD SegFormer feature 생성에서는 26,615 rows 중 26,607 rows가 실제 `segformer` backend를 사용했고, 8 rows만 dummy fallback을 사용했습니다.
- 전체 JAAD SegFormer CSV의 `road_pixel_ratio`는 평균 0.2329, 최소 0.000026, 최대 0.4521로 기록되었습니다.
- dummy/backend road relation feature 분포 비교는 `outputs/reports/road_relation_comparison.csv`에 저장됩니다.

### 3.5 Crossing classification 결과

확인 파일:

```text
outputs/results/baseline_results_crossing.csv
```

핵심 결과:

| feature_set | accuracy | precision | recall | f1 |
|---|---:|---:|---:|---:|
| bbox_only | 0.6043 | 0.6395 | 0.6733 | 0.6560 |
| pose_only | 0.6631 | 0.6847 | 0.7389 | 0.7108 |
| road_relation_only | 0.5653 | 0.5633 | 0.9975 | 0.7200 |
| pose_road_relation | 0.6626 | 0.6847 | 0.7376 | 0.7102 |
| pose_road_signal | 0.6794 | 0.7026 | 0.7419 | 0.7217 |

해석:

- `pose_road_signal`이 F1 0.7217로 가장 높습니다.
- 기존 `pose_road_relation` F1 0.7102보다 상승했습니다.
- 따라서 traffic signal context가 crossing classification에 추가 정보를 줄 가능성이 있습니다.

### 3.5.1 SegFormer road segmentation 적용 crossing 결과

확인 파일:

```text
outputs/results/baseline_results_crossing_segformer.csv
```

핵심 결과:

| feature_set | accuracy | precision | recall | f1 |
|---|---:|---:|---:|---:|
| bbox_only | 0.6043 | 0.6395 | 0.6733 | 0.6560 |
| pose_only | 0.6631 | 0.6847 | 0.7389 | 0.7108 |
| road_relation_only | 0.6260 | 0.6625 | 0.6778 | 0.6701 |
| pose_road_relation | 0.7217 | 0.7460 | 0.7632 | 0.7545 |
| pose_road_signal | 0.7281 | 0.7571 | 0.7580 | 0.7575 |

해석:

- dummy road mask 기준 `pose_road_relation` F1은 0.7102였지만, SegFormer road mask를 적용하면 F1이 0.7545로 상승했습니다.
- 같은 SegFormer CSV에서 `pose_only` F1은 0.7108이므로, 실제 road/sidewalk segmentation을 사용할 때 road relation feature가 crossing classification에 명확한 추가 정보를 제공합니다.
- `road_relation_only`는 dummy backend의 극단적인 recall 편향이 줄어들고 precision/recall이 더 균형적으로 바뀌었습니다.
- `pose_road_signal`은 F1 0.7575로 가장 높지만, signal feature는 traffic context의 annotation 한계를 함께 명시해야 합니다.

추가 진단 산출물:

```text
outputs/reports/rf_diagnostics/confusion_matrix_summary_segformer_label.csv
outputs/reports/rf_diagnostics/feature_importance_segformer_label.csv
outputs/reports/rf_diagnostics/feature_group_importance_segformer_label.csv
outputs/reports/rf_diagnostics/ablation_dummy_vs_segformer_label.csv
outputs/figures/rf_diagnostics/random_forest_segformer_pose_road_relation_label_confusion_matrix.png
```

진단 해석:

- `pose_only` confusion matrix는 FP 2,139, FN 1,642였습니다.
- SegFormer `pose_road_relation` confusion matrix는 FP 1,634, FN 1,489였습니다.
- 따라서 road relation feature를 결합하면 false positive와 false negative가 모두 줄었습니다.
- `pose_road_relation` feature importance에서 `distance_to_road`가 importance 0.1528로 1위였습니다.
- grouped importance 기준 `pose_road_relation` 모델은 pose 0.5308, road relation 0.2600, bbox position 0.2092로 구성되었습니다.

### 3.6 Risk prediction 결과

확인 파일:

```text
outputs/results/baseline_results_risk.csv
```

핵심 결과:

| feature_set | accuracy | precision | recall | f1 |
|---|---:|---:|---:|---:|
| bbox_only | 0.7623 | 0.9007 | 0.5620 | 0.6921 |
| pose_only | 0.7800 | 0.9924 | 0.5413 | 0.7005 |
| road_relation_only | 0.5246 | 0.0000 | 0.0000 | 0.0000 |
| pose_road_relation | 0.7721 | 0.9922 | 0.5248 | 0.6865 |
| pose_road_signal | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

주의:

- `pose_road_signal`의 risk F1 1.0은 일반적인 위험 예측 성능으로 해석하면 안 됩니다.
- `risk_label`이 `traffic_light_state`로부터 만들어졌고, `pose_road_signal`에도 `traffic_light_state_code`가 들어가기 때문입니다.
- 따라서 이 결과는 pipeline sanity check로는 유효하지만, 논문에서는 target-feature dependency 한계를 명시해야 합니다.

### 3.7 Signal presence split 결과

확인 파일:

```text
outputs/results/baseline_results_crossing_signal_present.csv
outputs/results/baseline_results_crossing_signal_absent.csv
outputs/results/baseline_results_signal_presence_f1_comparison.csv
```

핵심 결과:

| feature_set | signal_present f1 | signal_absent f1 |
|---|---:|---:|
| bbox_only | 0.6667 | 0.6607 |
| pose_only | 0.6912 | 0.7190 |
| road_relation_only | 0.6999 | 0.7214 |
| pose_road_relation | 0.6882 | 0.7212 |
| pose_road_signal | 0.6828 | 0.7216 |

해석:

- signal present subset은 1,938 rows로 작고 test video도 적습니다.
- 이 subset 결과는 전체 baseline을 대체하는 결과가 아니라, 신호등이 있는 장면에 대한 보조 분석입니다.

## 4. 코드에서 수정한 부분

| 파일 | 수정 내용 |
|---|---|
| `src/traffic_annotations.py` | JAAD traffic XML 분석, traffic_light 값 정규화, risk_label 생성 |
| `src/jaad_loader.py` | frame별 traffic annotation 로드 |
| `src/dataset_builder.py` | feature row에 traffic/risk 컬럼 추가 |
| `src/road_segmenter.py` | dummy/deeplabv3/segformer/yolo_seg road segmentation backend abstraction 및 SegFormer Cityscapes pretrained inference |
| `configs/default.yaml` | `pose_road_signal` feature set 추가 |
| `src/baseline_runner.py` | `--target-column label/risk_label` 기준 baseline 실행 |
| `src/train_random_forest.py` | `risk_label=-1` row 제외 후 학습 |
| `src/data_quality.py` | traffic/risk 분포 리포트 추가 |
| `scripts/create_analysis_screenshots.py` | 논문용 분석 과정 이미지 생성 |

## 5. 논문에 넣을 수 있는 설명 문장

아래 문장은 Method/Experiment 섹션에 사용할 수 있습니다.

```text
We do not train directly on raw video frames. Instead, each JAAD pedestrian instance is converted into a structured feature row containing bounding-box position, YOLO pose-derived features, road-relation features, and frame-level traffic-light context. The original JAAD crossing attribute is preserved as the crossing classification label. In addition, a weak signal-based risk_label is generated only when traffic-light state is available.
```

```text
The JAAD traffic annotation provides frame-level traffic_light values, including red and green states, but does not specify whether the signal is for vehicles or pedestrians. Therefore, the signal-based risk_label is treated as a weak conditional label rather than an independent ground-truth risk annotation.
```

```text
Adding traffic-light context improved the RandomForest crossing baseline from F1 0.7102 for pose_road_relation to F1 0.7217 for pose_road_signal. However, risk prediction results using pose_road_signal must be interpreted cautiously because risk_label is directly derived from traffic_light_state, which is also included as an input feature.
```

```text
When the heuristic dummy road mask was replaced with SegFormer Cityscapes road segmentation, the RandomForest pose_road_relation baseline improved from F1 0.7102 to F1 0.7545. This suggests that road-relation features are useful for crossing behavior recognition when the road mask captures actual scene geometry.
```

```text
The SegFormer-based pose_road_relation model reduced both false positives and false negatives compared with the pose-only baseline. Its RandomForest feature importance ranked distance_to_road as the most important individual feature, supporting the role of road geometry in the prediction.
```
