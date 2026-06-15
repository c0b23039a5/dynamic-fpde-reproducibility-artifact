"""ESC-50 metadata loading and deterministic splits."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class ESCSample:
    sample_id: str
    audio_path: Path
    filename: str
    fold: int
    category: str


@dataclass(frozen=True)
class ModeConfig:
    name: str
    max_classes: int | None
    max_train_per_class: int | None
    max_test_per_class: int | None


def get_mode_config(mode: str) -> ModeConfig:
    if mode == "smoke":
        return ModeConfig(mode, max_classes=5, max_train_per_class=5, max_test_per_class=2)
    if mode == "pilot":
        return ModeConfig(mode, max_classes=10, max_train_per_class=20, max_test_per_class=10)
    if mode == "full":
        return ModeConfig(mode, max_classes=None, max_train_per_class=None, max_test_per_class=None)
    raise ValueError("mode must be one of: smoke, pilot, full")


def read_esc50_metadata(dataset_root: str | Path) -> list[ESCSample]:
    root = Path(dataset_root)
    metadata_path = root / "meta" / "esc50.csv"
    audio_dir = root / "audio"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"ESC-50 metadata not found: {metadata_path}. Expected structure: "
            f"{root / 'audio'} and {root / 'meta' / 'esc50.csv'}."
        )
    if not audio_dir.exists():
        raise FileNotFoundError(
            f"ESC-50 audio directory not found: {audio_dir}. Expected structure: "
            f"{root / 'audio'} and {root / 'meta' / 'esc50.csv'}."
        )
    rows: list[ESCSample] = []
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"filename", "fold", "category"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"ESC-50 metadata missing required columns: {sorted(missing)}")
        for row in reader:
            filename = str(row["filename"])
            rows.append(
                ESCSample(
                    sample_id=Path(filename).stem,
                    audio_path=audio_dir / filename,
                    filename=filename,
                    fold=int(row["fold"]),
                    category=str(row["category"]),
                )
            )
    if not rows:
        raise ValueError(f"ESC-50 metadata contains no samples: {metadata_path}")
    return rows


def _stable_categories(samples: Sequence[ESCSample]) -> list[str]:
    return sorted({sample.category for sample in samples})


def _limit_per_class(samples: Sequence[ESCSample], limit: int | None) -> list[ESCSample]:
    if limit is None:
        return list(samples)
    out: list[ESCSample] = []
    counts: dict[str, int] = {}
    for sample in sorted(samples, key=lambda item: (item.category, item.sample_id)):
        count = counts.get(sample.category, 0)
        if count < limit:
            out.append(sample)
            counts[sample.category] = count + 1
    return out


def stratified_validation_split(
    train_samples: Sequence[ESCSample],
    *,
    seed: int,
    val_per_class: int = 1,
) -> tuple[list[ESCSample], list[ESCSample]]:
    rng = np.random.default_rng(seed)
    train_out: list[ESCSample] = []
    val_out: list[ESCSample] = []
    for category in _stable_categories(train_samples):
        group = [sample for sample in train_samples if sample.category == category]
        order = np.arange(len(group))
        rng.shuffle(order)
        take = min(val_per_class, max(0, len(group) - 1))
        val_indices = set(order[:take].tolist())
        for idx, sample in enumerate(group):
            if idx in val_indices:
                val_out.append(sample)
            else:
                train_out.append(sample)
    return train_out, val_out


def split_esc50(
    samples: Sequence[ESCSample],
    *,
    fold: int,
    mode_config: ModeConfig,
    seed: int,
) -> tuple[list[ESCSample], list[ESCSample], list[ESCSample]]:
    categories = _stable_categories(samples)
    if mode_config.max_classes is not None:
        categories = categories[: mode_config.max_classes]
    selected = set(categories)
    filtered = [sample for sample in samples if sample.category in selected]
    train_candidates = [sample for sample in filtered if sample.fold != fold]
    test_candidates = [sample for sample in filtered if sample.fold == fold]
    if not train_candidates:
        raise ValueError(f"no ESC-50 training samples found for fold {fold}")
    if not test_candidates:
        raise ValueError(f"no ESC-50 test samples found for fold {fold}")
    limited_train = _limit_per_class(train_candidates, mode_config.max_train_per_class)
    limited_test = _limit_per_class(test_candidates, mode_config.max_test_per_class)
    train_split, val_split = stratified_validation_split(limited_train, seed=seed)
    if not val_split:
        val_split = train_split[:1]
        train_split = train_split[1:] or val_split
    return train_split, val_split, limited_test


def parse_folds(fold: int | None, folds: str | None) -> list[int]:
    if folds:
        parsed = [int(part.strip()) for part in folds.split(",") if part.strip()]
        if not parsed:
            raise ValueError("--folds did not contain any fold ids")
        return parsed
    return [1 if fold is None else int(fold)]


def labels_for(samples: Iterable[ESCSample]) -> list[str]:
    return [sample.category for sample in samples]
