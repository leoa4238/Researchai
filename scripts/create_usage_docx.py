from __future__ import annotations

import html
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "results" / "jaywalking-risk-recognition_usage_guide.docx"


def p(text: str = "", style: str | None = None) -> str:
    """간단한 Word paragraph XML을 생성합니다."""
    style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{style_xml}<w:r><w:t>{html.escape(text)}</w:t></w:r></w:p>"


def code(text: str) -> str:
    """명령어 블록처럼 보이도록 Courier New 글꼴의 paragraph를 생성합니다."""
    escaped = html.escape(text)
    return (
        "<w:p><w:pPr><w:spacing w:before=\"80\" w:after=\"80\"/>"
        "<w:shd w:fill=\"F2F2F2\"/></w:pPr>"
        "<w:r><w:rPr><w:rFonts w:ascii=\"Courier New\" w:hAnsi=\"Courier New\"/>"
        "<w:sz w:val=\"20\"/></w:rPr>"
        f"<w:t xml:space=\"preserve\">{escaped}</w:t></w:r></w:p>"
    )


def bullet(text: str) -> str:
    return p(f"- {text}")


def document_xml() -> str:
    body = []
    body.append(p("무단횡단 위험 행동 인식 AI 연구 개발환경 사용법", "Title"))
    body.append(p("연구 주제: 도로-인도 공간 관계와 보행자 포즈 정보를 결합한 무단횡단 위험 행동 인식 모델"))
    body.append(p("작성일: 2026년 4월 28일"))

    body.append(p("1. 프로젝트 개요", "Heading1"))
    body.append(p("이 프로젝트는 JAAD, PIE 등 보행자 영상 데이터셋을 입력으로 받아 보행자 검출, 포즈 추정, 도로/인도 segmentation, 공간 관계 feature 추출, CSV 데이터셋 생성, 위험 행동 분류 모델 학습까지 이어지는 연구용 skeleton입니다."))
    body.append(bullet("Pose-only feature와 Pose+Road-relation feature를 비교 실험할 수 있습니다."))
    body.append(bullet("데이터셋이 없어도 dummy CSV를 자동 생성하여 학습 코드 실행을 확인할 수 있습니다."))
    body.append(bullet("Windows PowerShell 환경에서 실행 가능하도록 구성했습니다."))

    body.append(p("2. 폴더 구조", "Heading1"))
    body.append(code("""jaywalking-risk-recognition/
  data/
    raw/ annotations/ videos/ processed/ features/
  models/
    pose/ segmentation/ classifiers/
  src/
    config.py, video_reader.py, pedestrian_detector.py, pose_extractor.py
    road_segmenter.py, feature_extractor.py, dataset_builder.py
    train_random_forest.py, train_lstm.py, evaluate.py, visualize.py
  configs/default.yaml
  notebooks/
  outputs/logs, outputs/figures, outputs/results
  requirements.txt
  README.md
  main.py"""))

    body.append(p("3. 개발환경 설치", "Heading1"))
    body.append(p("Python 3.10 이상을 권장합니다. 현재 PC에서 python 명령이 3.9를 가리킬 수 있으므로, Windows에서는 py 런처로 버전을 지정하는 방식이 안전합니다."))
    body.append(code("""cd D:\\reserch\\jaywalking-risk-recognition
py -3.10 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt"""))
    body.append(p("이미 생성된 가상환경을 사용할 경우에는 아래 명령으로 활성화합니다."))
    body.append(code("""cd D:\\reserch\\jaywalking-risk-recognition
.\\.venv\\Scripts\\Activate.ps1"""))

    body.append(p("4. 설정 파일", "Heading1"))
    body.append(p("모든 경로와 주요 파라미터는 configs/default.yaml에서 관리합니다. 입력 영상 경로, feature CSV 저장 위치, YOLO 모델명, segmentation backend, 학습 epoch, batch size, feature set 등을 이 파일에서 수정합니다."))
    body.append(bullet("paths.input_video: 기본 입력 영상 경로"))
    body.append(bullet("paths.feature_csv: 실제 영상에서 추출한 feature CSV 저장 경로"))
    body.append(bullet("paths.dummy_feature_csv: dummy 학습용 CSV 저장 경로"))
    body.append(bullet("experiments.feature_sets.pose_only: pose 좌표 중심 feature"))
    body.append(bullet("experiments.feature_sets.pose_road_relation: pose와 도로 관계를 결합한 feature"))

    body.append(p("5. Dummy 데이터로 전체 동작 확인", "Heading1"))
    body.append(p("데이터셋이 아직 없어도 아래 명령으로 dummy CSV 생성, RandomForest 학습, LSTM skeleton 학습, 평가 결과 저장까지 확인할 수 있습니다."))
    body.append(code("python main.py --mode dummy"))
    body.append(p("생성 결과 위치는 다음과 같습니다."))
    body.append(bullet("data/features/dummy_features.csv"))
    body.append(bullet("models/classifiers/random_forest_pose_road_relation.joblib"))
    body.append(bullet("models/classifiers/lstm_pose_road_relation.pt"))
    body.append(bullet("outputs/results/*_metrics.json"))
    body.append(bullet("outputs/figures/*_confusion_matrix.png"))

    body.append(p("6. 실제 영상으로 Feature CSV 생성", "Heading1"))
    body.append(p("영상 파일을 data/videos 폴더에 넣은 뒤 아래처럼 실행합니다."))
    body.append(code("python main.py --mode features --video data/videos/sample.mp4"))
    body.append(p("실행 흐름은 VideoReader -> PedestrianDetector -> PoseExtractor -> RoadSegmenter -> FeatureExtractor -> CSV 저장 순서입니다."))

    body.append(p("7. 모델 학습", "Heading1"))
    body.append(p("RandomForest만 학습하려면 다음 명령을 사용합니다."))
    body.append(code("python main.py --mode train-rf --feature-set pose_road_relation"))
    body.append(p("LSTM skeleton만 학습하려면 다음 명령을 사용합니다."))
    body.append(code("python main.py --mode train-lstm --feature-set pose_road_relation"))
    body.append(p("영상 feature 생성부터 두 모델 학습까지 한 번에 실행하려면 다음 명령을 사용합니다."))
    body.append(code("python main.py --mode all --video data/videos/sample.mp4 --feature-set pose_road_relation"))

    body.append(p("8. 비교 실험 방법", "Heading1"))
    body.append(p("Pose-only 모델과 Pose+Road-relation 모델 비교는 --feature-set 옵션으로 수행합니다."))
    body.append(code("""python main.py --mode dummy --feature-set pose_only
python main.py --mode dummy --feature-set pose_road_relation"""))
    body.append(p("결과 JSON 파일의 accuracy, precision, recall, f1, confusion_matrix를 비교하면 됩니다."))

    body.append(p("9. Feature CSV 컬럼", "Heading1"))
    for column in [
        "frame_id: 프레임 번호",
        "pedestrian_id: 보행자 ID",
        "center_x, center_y: 보행자 bbox 중심 좌표",
        "left_ankle_x, left_ankle_y: 왼쪽 발목 좌표",
        "right_ankle_x, right_ankle_y: 오른쪽 발목 좌표",
        "body_direction: 어깨 중심에서 골반 중심으로 향하는 각도",
        "step_direction: 양쪽 발목 위치로 계산한 보행 방향 proxy",
        "distance_to_road: 보행자 중심에서 가장 가까운 도로 mask까지의 거리",
        "foot_on_road: 발목 keypoint가 도로 위에 있는지 여부",
        "center_on_road: 보행자 중심이 도로 위에 있는지 여부",
        "approach_rate: 이전 프레임 대비 도로까지의 거리 감소량",
        "label: 위험 행동 정답 라벨",
    ]:
        body.append(bullet(column))

    body.append(p("10. 현재 Skeleton의 한계와 다음 구현 단계", "Heading1"))
    body.append(bullet("YOLO 모델이 설치되어 있으면 ultralytics YOLO를 로드하고, 실패하면 dummy detector/pose를 사용합니다."))
    body.append(bullet("segmentation은 현재 dummy backend가 기본이며, 하단 영역을 road로 가정합니다. 실제 road/sidewalk 모델 checkpoint가 준비되면 src/road_segmenter.py를 교체합니다."))
    body.append(bullet("JAAD/PIE annotation parser는 아직 skeleton에 포함되지 않았습니다. annotation label을 label 컬럼에 연결하는 단계가 다음 작업입니다."))
    body.append(bullet("LSTM은 실행 가능한 최소 학습 루프이며, 실제 연구에서는 train/validation split, normalization, class imbalance 처리, sequence sampling 전략을 보강하는 것이 좋습니다."))

    body.append(p("11. 문제 해결", "Heading1"))
    body.append(bullet("ModuleNotFoundError가 발생하면 가상환경 활성화 후 pip install -r requirements.txt를 다시 실행합니다."))
    body.append(bullet("입력 영상이 없다는 오류가 나오면 --video 옵션 경로를 확인하거나 configs/default.yaml의 paths.input_video를 수정합니다."))
    body.append(bullet("Python 버전이 3.10 미만이면 3.10 이상 Python을 설치한 뒤 가상환경을 다시 생성합니다."))
    body.append(bullet("Windows에서 matplotlib GUI 관련 오류가 나지 않도록 코드에서 Agg backend를 사용하도록 설정했습니다."))

    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {''.join(body)}
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/>
    </w:sectPr>
  </w:body>
</w:document>"""


def styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:rPr><w:rFonts w:ascii="Malgun Gothic" w:hAnsi="Malgun Gothic" w:eastAsia="Malgun Gothic"/><w:sz w:val="21"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:rPr><w:b/><w:rFonts w:ascii="Malgun Gothic" w:hAnsi="Malgun Gothic" w:eastAsia="Malgun Gothic"/><w:sz w:val="34"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:rPr><w:b/><w:rFonts w:ascii="Malgun Gothic" w:hAnsi="Malgun Gothic" w:eastAsia="Malgun Gothic"/><w:sz w:val="28"/></w:rPr>
  </w:style>
</w:styles>"""


def create_docx() -> Path:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
    doc_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""

    with zipfile.ZipFile(OUTPUT_PATH, "w", compression=zipfile.ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types)
        docx.writestr("_rels/.rels", rels)
        docx.writestr("word/_rels/document.xml.rels", doc_rels)
        docx.writestr("word/document.xml", document_xml())
        docx.writestr("word/styles.xml", styles_xml())

    return OUTPUT_PATH


if __name__ == "__main__":
    print(create_docx())
