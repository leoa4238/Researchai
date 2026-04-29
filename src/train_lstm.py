from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import Config
from src.data_quality import official_train_test_masks, validate_before_training, warn_if_split_has_single_label
from src.evaluate import classification_metrics
from src.experiment_summary import write_experiment_summary


class RiskLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(inputs)
        return self.classifier(hidden[-1])


def make_sequences(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    group_columns = ["video_id", "pedestrian_id"] if "video_id" in data.columns else ["pedestrian_id"]
    data = data.sort_values([*group_columns, "frame_id"]).reset_index(drop=True)
    sequences: list[np.ndarray] = []
    labels: list[int] = []
    stats = {"tracks": 0, "excluded_short_tracks": 0}

    for _, group in data.groupby(group_columns):
        stats["tracks"] += 1
        values = group[feature_columns].to_numpy(dtype=np.float32)
        targets = group[target_column].astype(int).to_numpy()
        if len(group) < sequence_length:
            stats["excluded_short_tracks"] += 1
            continue
        for start in range(0, len(group) - sequence_length + 1):
            end = start + sequence_length
            sequences.append(values[start:end])
            labels.append(int(targets[end - 1]))

    if not sequences:
        raise ValueError("No LSTM sequences were created. Check row counts and sequence_length.")
    return np.stack(sequences), np.array(labels, dtype=np.int64), stats


def train_lstm(
    config: Config,
    csv_path: str | Path,
    feature_set_name: str = "pose_road_relation",
    split_strategy: str | None = None,
) -> dict[str, float | int | list[list[int]]]:
    data = pd.read_csv(csv_path)
    target_column = config.get("training.target_column", "label")
    feature_columns = config.get(f"experiments.feature_sets.{feature_set_name}")
    if feature_columns is None:
        raise KeyError(f"Unknown feature set: {feature_set_name}")

    split_strategy = split_strategy or config.get("training.split_strategy", "official")
    validate_before_training(data, feature_columns, target_column, split_strategy)

    sequence_length = config.get("training.sequence_length", 8)
    if split_strategy == "official":
        train_mask, test_mask = official_train_test_masks(data)
        warn_if_split_has_single_label(data, target_column, train_mask, test_mask)
        x_train, y_train, train_sequence_stats = make_sequences(data.loc[train_mask], feature_columns, target_column, sequence_length)
        x_test, y_test, test_sequence_stats = make_sequences(data.loc[test_mask], feature_columns, target_column, sequence_length)
    elif split_strategy == "random":
        print("[split] using random train/test split")
        x, y, all_sequence_stats = make_sequences(data, feature_columns, target_column, sequence_length)
        split_index = max(1, int(len(x) * (1 - config.get("training.test_size", 0.2))))
        x_train, x_test = x[:split_index], x[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        if len(x_test) == 0:
            x_test, y_test = x_train, y_train
        train_sequence_stats = {"tracks": all_sequence_stats["tracks"], "excluded_short_tracks": all_sequence_stats["excluded_short_tracks"]}
        test_sequence_stats = {"tracks": 0, "excluded_short_tracks": 0}
    else:
        raise ValueError(f"Unknown split strategy: {split_strategy}")

    _print_sequence_stats("train", y_train, train_sequence_stats, len(x_train))
    _print_sequence_stats("test", y_test, test_sequence_stats, len(x_test))

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
    metrics["train_rows"] = int(len(x_train))
    metrics["test_rows"] = int(len(x_test))
    metrics["train_sequences"] = int(len(x_train))
    metrics["test_sequences"] = int(len(x_test))
    metrics["train_excluded_short_tracks"] = int(train_sequence_stats["excluded_short_tracks"])
    metrics["test_excluded_short_tracks"] = int(test_sequence_stats["excluded_short_tracks"])

    model_path = config.path("paths.classifier_model_dir") / f"lstm_{feature_set_name}.pt"
    result_path = config.path("paths.result_dir") / f"lstm_{feature_set_name}_metrics.json"
    csv_result_path = config.path("paths.result_dir") / "lstm_results.csv"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    result_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "feature_set": feature_set_name,
                "model": "LSTM",
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "train_rows": metrics["train_rows"],
                "test_rows": metrics["test_rows"],
                "train_sequences": metrics["train_sequences"],
                "test_sequences": metrics["test_sequences"],
                "train_excluded_short_tracks": metrics["train_excluded_short_tracks"],
                "test_excluded_short_tracks": metrics["test_excluded_short_tracks"],
            }
        ]
    ).to_csv(csv_result_path, index=False, encoding="utf-8-sig")
    print(f"[lstm] results saved: {csv_result_path}")
    write_experiment_summary(config)

    return metrics


def _print_sequence_stats(split: str, labels: np.ndarray, stats: dict[str, int], sequence_count: int) -> None:
    values, counts = np.unique(labels, return_counts=True)
    label_distribution = {int(value): int(count) for value, count in zip(values, counts)}
    print(
        f"[LSTM sequence] {split}: sequences={sequence_count}, "
        f"label_distribution={label_distribution}, "
        f"excluded_short_tracks={stats['excluded_short_tracks']}, tracks={stats['tracks']}"
    )
