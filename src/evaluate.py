from __future__ import annotations

import os
from pathlib import Path

# Windows 권한 문제를 피하기 위해 matplotlib 설정/cache를 프로젝트 내부에 둡니다.
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / "matplotlib_cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | list[list[int]]]:
    """Accuracy, Precision, Recall, F1, Confusion Matrix를 계산합니다."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).astype(int).tolist(),
    }


def save_confusion_matrix_figure(y_true: np.ndarray, y_pred: np.ndarray, output_path: str | Path) -> None:
    """혼동행렬을 이미지로 저장합니다."""
    matrix = confusion_matrix(y_true, y_pred)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, str(value), ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
