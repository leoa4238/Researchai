from __future__ import annotations

import html
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "results" / "jaywalking-risk-recognition_progress_report.docx"


def paragraph(text: str = "", style: str | None = None) -> str:
    style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{style_xml}<w:r><w:t>{html.escape(text)}</w:t></w:r></w:p>"


def bullet(text: str) -> str:
    return paragraph(f"- {text}")


def code_block(text: str) -> str:
    escaped = html.escape(text)
    return (
        '<w:p><w:pPr><w:spacing w:before="80" w:after="80"/>'
        '<w:shd w:fill="F2F2F2"/></w:pPr>'
        '<w:r><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/>'
        '<w:sz w:val="19"/></w:rPr>'
        f'<w:t xml:space="preserve">{escaped}</w:t></w:r></w:p>'
    )


def document_xml() -> str:
    body: list[str] = []
    body.append(paragraph("무단횡단 위험 인식 연구 프로젝트 진행 보고서", "Title"))
    body.append(paragraph("작성일: 2026년 4월 29일"))
    body.append(paragraph("프로젝트 경로: D:\\reserch\\jaywalking-risk-recognition"))

    body.append(paragraph("1. 현재 진행상황 요약", "Heading1"))
    body.append(bullet("JAAD 영상 클립 다운로드 및 압축 해제 완료"))
    body.append(bullet("data/raw/JAAD/JAAD_clips 폴더에 video_0001.mp4 형식의 영상 346개 준비 완료"))
    body.append(bullet("JAAD 2.0 annotation GitHub 저장소 clone 완료"))
    body.append(bullet("data/raw/JAAD_annotations 폴더에 annotations, attributes, traffic, vehicle, split_ids, jaad_data.py 준비 완료"))
    body.append(bullet("JAAD XML annotation을 읽는 src/jaad_loader.py 모듈 추가"))
    body.append(bullet("JAAD bbox와 cross label을 기존 feature 추출 파이프라인에 연결"))
    body.append(bullet("video_0002 기준 JAAD feature CSV 생성 및 RandomForest/LSTM 학습 실행 확인"))

    body.append(paragraph("2. 데이터셋 구성", "Heading1"))
    body.append(paragraph("현재 준비된 데이터는 영상과 annotation이 분리되어 있습니다. 영상은 JAAD_clips 폴더에 있고, annotation은 JAAD_annotations 폴더에 있습니다."))
    body.append(code_block("""data/raw/JAAD/
  JAAD_clips/
    video_0001.mp4
    video_0002.mp4
    ...
  JAAD_clips.zip

data/raw/JAAD_annotations/
  annotations/
  annotations_appearance/
  annotations_attributes/
  annotations_traffic/
  annotations_vehicle/
  split_ids/
  jaad_data.py"""))
    body.append(bullet("영상 클립 수: 346개"))
    body.append(bullet("annotation XML 수: 346개"))
    body.append(bullet("현재 feature 생성 검증에 사용한 영상: video_0002"))

    body.append(paragraph("3. 전체 동작 방식", "Heading1"))
    body.append(paragraph("현재 파이프라인은 JAAD annotation의 보행자 bbox를 ground truth detector처럼 사용합니다. 즉, 처음부터 YOLO 보행자 검출 성능에 의존하지 않고 JAAD가 제공하는 정답 bbox와 crossing label을 이용해 feature CSV를 만듭니다."))
    body.append(code_block("""JAAD XML annotation
  -> frame별 pedestrian bbox 추출
  -> cross 속성을 label로 변환
  -> bbox를 Detection 객체로 변환
  -> dummy pose 또는 YOLO pose 추출
  -> road/sidewalk segmentation
  -> pose + road relation feature 계산
  -> data/features/jaad_features.csv 저장
  -> RandomForest 또는 LSTM 학습"""))
    body.append(paragraph("JAAD 모드에서는 기본적으로 pose_backend를 dummy로 설정했습니다. 현재 환경에서 YOLO pose 가중치 다운로드가 네트워크 제한으로 실패했기 때문입니다. 그래도 bbox 기반 dummy pose를 사용하면 feature 생성과 학습 파이프라인 검증은 가능합니다."))

    body.append(paragraph("4. 주요 코드 변경사항", "Heading1"))
    body.append(bullet("src/jaad_loader.py: JAAD XML 파일을 읽고 frame별 JaadBox 데이터로 변환"))
    body.append(bullet("src/dataset_builder.py: build_jaad_feature_dataset() 추가"))
    body.append(bullet("src/feature_extractor.py: annotation label을 feature row의 label 컬럼으로 주입 가능하게 수정"))
    body.append(bullet("main.py: jaad-features 모드와 --csv-path, --jaad-video-id, --limit-videos 옵션 추가"))
    body.append(bullet("configs/default.yaml: JAAD annotation/video/feature 경로와 JAAD용 pose backend 설정 추가"))
    body.append(bullet("src/pose_extractor.py, src/pedestrian_detector.py: Ultralytics 설정 폴더를 프로젝트 내부 outputs/ultralytics로 지정"))

    body.append(paragraph("5. 생성되는 Feature CSV", "Heading1"))
    body.append(paragraph("JAAD feature CSV는 다음 위치에 저장됩니다."))
    body.append(code_block("data/features/jaad_features.csv"))
    body.append(paragraph("현재 생성된 주요 컬럼은 다음과 같습니다."))
    for item in [
        "video_id: JAAD 영상 ID",
        "source_pedestrian_id: JAAD 원본 pedestrian ID",
        "action: standing 또는 walking",
        "look: looking 또는 not-looking",
        "occlusion: none, part, full",
        "frame_id: 프레임 번호",
        "pedestrian_id: 내부 숫자형 pedestrian ID",
        "center_x, center_y: bbox 중심 좌표",
        "left_ankle_x, left_ankle_y, right_ankle_x, right_ankle_y: pose 기반 발목 좌표",
        "body_direction, step_direction: 신체 방향과 보행 방향 proxy",
        "distance_to_road: 보행자 중심에서 road mask까지의 거리",
        "foot_on_road, center_on_road: 발 또는 중심이 도로 위에 있는지 여부",
        "approach_rate: 이전 프레임 대비 도로와의 거리 변화량",
        "label: JAAD cross 속성 기반 label, crossing이면 1, not-crossing이면 0",
    ]:
        body.append(bullet(item))

    body.append(paragraph("6. 사용 방법", "Heading1"))
    body.append(paragraph("PowerShell에서 프로젝트 루트로 이동한 뒤, 가상환경 Python을 직접 사용합니다."))
    body.append(code_block("""cd D:\\reserch\\jaywalking-risk-recognition
.\\.venv\\Scripts\\python.exe main.py --mode jaad-features --jaad-video-id video_0002 --limit-videos 1"""))
    body.append(paragraph("위 명령은 video_0002 하나를 읽어서 data/features/jaad_features.csv를 생성합니다."))
    body.append(paragraph("생성된 JAAD feature CSV로 RandomForest를 학습하려면 다음 명령을 사용합니다."))
    body.append(code_block(""".\\.venv\\Scripts\\python.exe main.py --mode train-rf --csv-path data\\features\\jaad_features.csv --feature-set pose_road_relation"""))
    body.append(paragraph("LSTM을 학습하려면 다음 명령을 사용합니다."))
    body.append(code_block(""".\\.venv\\Scripts\\python.exe main.py --mode train-lstm --csv-path data\\features\\jaad_features.csv --feature-set pose_road_relation"""))
    body.append(paragraph("여러 JAAD 영상을 처리하려면 --limit-videos 값을 늘립니다. 설정 파일의 jaad.limit_videos 기본값은 현재 1입니다."))
    body.append(code_block(""".\\.venv\\Scripts\\python.exe main.py --mode jaad-features --limit-videos 10"""))
    body.append(paragraph("전체 영상을 처리하려면 설정 파일의 jaad.limit_videos를 비우거나, 코드에서 limit 제한을 제거한 뒤 실행하는 방식이 안전합니다."))

    body.append(paragraph("7. 검증 결과", "Heading1"))
    body.append(paragraph("video_0002 하나로 JAAD feature 생성과 학습 파이프라인을 검증했습니다. 단일 영상 기준 결과이므로 모델 성능 해석보다는 파이프라인 연결 확인용으로 보는 것이 맞습니다."))
    body.append(bullet("생성 row 수: 69"))
    body.append(bullet("label 분포: label 1이 38개, label 0이 31개"))
    body.append(bullet("RandomForest accuracy: 0.9286"))
    body.append(bullet("RandomForest f1: 0.9412"))
    body.append(bullet("LSTM accuracy: 0.8182"))
    body.append(bullet("LSTM f1: 0.0"))
    body.append(paragraph("LSTM은 데이터가 너무 적고 단일 영상만 사용했기 때문에 현재 성능이 낮게 나온 것으로 보입니다. 전체 데이터셋과 적절한 sequence split을 적용해야 의미 있는 평가가 가능합니다."))

    body.append(paragraph("8. 현재 한계와 다음 작업", "Heading1"))
    body.append(bullet("현재 road segmentation은 dummy 방식입니다. 화면 하단 영역을 도로로 가정하므로 실제 도로/인도 구분 정확도는 제한적입니다."))
    body.append(bullet("JAAD 모드는 현재 dummy pose를 사용합니다. YOLO pose 가중치를 준비하면 실제 keypoint 기반 feature로 바꿀 수 있습니다."))
    body.append(bullet("전체 346개 영상 처리 후 train/validation/test split을 JAAD split_ids와 연결해야 합니다."))
    body.append(bullet("현재 label은 cross 속성만 사용합니다. 이후 action, look, traffic, vehicle annotation을 결합하면 위험 행동 정의를 더 정교하게 만들 수 있습니다."))
    body.append(bullet("README와 일부 기존 코드 주석은 한글 인코딩이 깨져 있어 정리할 필요가 있습니다."))

    body.append(paragraph("9. 추천 다음 단계", "Heading1"))
    body.append(bullet("JAAD 전체 또는 일정 수의 영상으로 jaad_features.csv 생성"))
    body.append(bullet("JAAD split_ids를 이용해 train/val/test 분할 적용"))
    body.append(bullet("road segmentation을 실제 road/sidewalk mask 또는 외부 segmentation 모델로 교체"))
    body.append(bullet("YOLO pose 가중치를 사전 다운로드하거나 로컬 모델 경로를 설정"))
    body.append(bullet("RandomForest baseline 성능을 먼저 안정화한 뒤 LSTM sequence 구성 개선"))

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
    <w:rPr>
      <w:rFonts w:ascii="Malgun Gothic" w:hAnsi="Malgun Gothic" w:eastAsia="Malgun Gothic"/>
      <w:sz w:val="21"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:rPr>
      <w:b/>
      <w:rFonts w:ascii="Malgun Gothic" w:hAnsi="Malgun Gothic" w:eastAsia="Malgun Gothic"/>
      <w:sz w:val="34"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:rPr>
      <w:b/>
      <w:rFonts w:ascii="Malgun Gothic" w:hAnsi="Malgun Gothic" w:eastAsia="Malgun Gothic"/>
      <w:sz w:val="28"/>
    </w:rPr>
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
