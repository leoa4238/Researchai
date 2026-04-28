from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import Config
from src.evaluate import classification_metrics


class RiskLSTM(nn.Module):
    """시계열 pose/road-relation feature를 입력받는 LSTM 분류기 skeleton입니다."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(inputs)
        last_hidden = hidden[-1]
        return self.classifier(last_hidden)


def make_sequences(data: pd.DataFrame, feature_columns: list[str], target_column: str, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    """frame 순서대로 고정 길이 sequence를 만들어 LSTM 입력 형태로 변환합니다."""
    data = data.sort_values(["pedestrian_id", "frame_id"]).reset_index(drop=True)
    sequences: list[np.ndarray] = []
    labels: list[int] = []

    for _, group in data.groupby("pedestrian_id"):
        values = group[feature_columns].fillna(0).to_numpy(dtype=np.float32)
        targets = group[target_column].astype(int).to_numpy()
        if len(group) < sequence_length:
            continue

        for start in range(0, len(group) - sequence_length + 1):
            end = start + sequence_length
            sequences.append(values[start:end])
            labels.append(int(targets[end - 1]))

    if not sequences:
        raise ValueError("LSTM sequence를 만들 수 없습니다. CSV row 수나 sequence_length를 확인하세요.")

    return np.stack(sequences), np.array(labels, dtype=np.int64)


def train_lstm(config: Config, csv_path: str | Path, feature_set_name: str = "pose_road_relation") -> dict[str, float | list[list[int]]]:
    """PyTorch LSTM skeleton 학습 루프입니다. 작은 dummy CSV에서도 실행됩니다."""
    data = pd.read_csv(csv_path)
    target_column = config.get("training.target_column", "label")
    feature_columns = config.get(f"experiments.feature_sets.{feature_set_name}")
    if feature_columns is None:
        raise KeyError(f"알 수 없는 feature set입니다: {feature_set_name}")

    sequence_length = config.get("training.sequence_length", 8)
    x, y = make_sequences(data, feature_columns, target_column, sequence_length)

    split_index = max(1, int(len(x) * (1 - config.get("training.test_size", 0.2))))
    x_train, x_test = x[:split_index], x[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    if len(x_test) == 0:
        x_test, y_test = x_train, y_train

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=config.get("training.batch_size", 16),
        shuffle=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RiskLSTM(
        input_size=len(feature_columns),
        hidden_size=config.get("training.lstm_hidden_size", 64),
        num_layers=config.get("training.lstm_num_layers", 1),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("training.learning_rate", 0.001))
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(config.get("training.epochs", 3)):
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x_test).to(device))
        predictions = torch.argmax(logits, dim=1).cpu().numpy()

    metrics = classification_metrics(y_test, predictions)

    model_path = config.path("paths.classifier_model_dir") / f"lstm_{feature_set_name}.pt"
    result_path = config.path("paths.result_dir") / f"lstm_{feature_set_name}_metrics.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    result_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    return metrics
