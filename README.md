# Jaywalking Risk Recognition

도로-인도 공간 관계와 보행자 포즈 정보를 결합해 무단횡단 위험 행동을 인식하는 Python 연구 개발환경 skeleton입니다.

## 주요 기능

- 영상 frame 단위 읽기
- YOLOv8/YOLOv11 기반 보행자 검출 skeleton
- YOLO-Pose 기반 pose keypoint 추출 skeleton
- road/sidewalk segmentation skeleton
- pose 좌표와 road mask의 공간 관계 feature CSV 생성
- RandomForest 학습
- PyTorch LSTM 학습 skeleton
- pose-only 모델과 pose+road-relation 모델 비교 실험

## 설치

Python 3.10 이상을 권장합니다. Windows에서 여러 Python이 설치되어 있으면 `py -3.10` 또는 `py -3.12`처럼 버전을 지정해 가상환경을 만드세요.

```powershell
cd jaywalking-risk-recognition
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 빠른 실행

데이터셋이 없어도 dummy CSV를 생성하고 RandomForest/LSTM 학습까지 실행합니다.

```powershell
python main.py --mode dummy
```

실제 영상 feature CSV 생성:

```powershell
python main.py --mode features --video data/videos/sample.mp4
```

전체 파이프라인:

```powershell
python main.py --mode all --video data/videos/sample.mp4 --feature-set pose_road_relation
```

Pose-only 비교 실험:

```powershell
python main.py --mode dummy --feature-set pose_only
```

## 설정

모든 경로와 주요 파라미터는 `configs/default.yaml`에서 관리합니다.

## 생성되는 feature 컬럼

- frame_id
- pedestrian_id
- center_x, center_y
- left_ankle_x, left_ankle_y
- right_ankle_x, right_ankle_y
- body_direction
- step_direction
- distance_to_road
- foot_on_road
- center_on_road
- approach_rate
- label

## 참고

현재 segmentation은 학습된 모델이 없을 때 하단 영역을 road로 가정하는 dummy mask를 반환합니다. JAAD/PIE annotation과 segmentation label mapping이 준비되면 `src/road_segmenter.py`의 model loading과 inference 부분을 교체하면 됩니다.
