"""Canonical Yeom membership-inference evaluation for federated unlearning.

This module compares a pretrained checkpoint with an unlearned checkpoint.  It
does not train an attack model, a shadow model, or another target model.  The
pretrained checkpoint is read from ``RESUME`` in an unlearning environment and
the unlearned checkpoint is resolved from the same environment's generated
save path.

Run it from the repository root using the same environment convention as the
federated client and server:

    $env:EXP_ENV_DIR = "path/to/unlearning/env"
    python -m src.attacks.yeom

By default the evaluator uses clean images and original ground-truth labels,
writes auditable CSV/JSON results below the unlearned run directory, and pushes
the paired pretrained/unlearned metrics and result artifacts to Weights &
Biases.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from ..dataloaders.client_dataloader import (
    CIFAR100_NORMALIZED_WHITE,
    Caltech101Wrapper,
    _cifar100_eval_transform,
)
from ..dataloaders.transformers_utils import _add_square_trigger
from ..utils.utils import generate_save_path, load_config, load_model


MODEL_PRETRAINED = "pretrained"
MODEL_UNLEARNED = "unlearned"
SPLIT_FORGET = "forget"
SPLIT_RETAIN = "retain"
SPLIT_NONMEMBER = "non-member"
SPLIT_ORDER = (SPLIT_FORGET, SPLIT_RETAIN, SPLIT_NONMEMBER)
MODEL_ORDER = (MODEL_PRETRAINED, MODEL_UNLEARNED)

REQUIRED_INDEX_KEYS = {
    "training_set",
    "full_training",
    "forget",
    "val",
    "test",
    "retrain",
}


@dataclass(frozen=True)
class SampleRef:
    """A stable reference to one source-dataset sample."""

    client_id: int
    source_index: int


@dataclass(frozen=True)
class PartitionIndices:
    """Saved indices for one federated client partition."""

    client_id: int
    path: Path
    training_set: np.ndarray
    full_training: np.ndarray
    forget: np.ndarray
    val: np.ndarray
    test: np.ndarray
    retrain: np.ndarray

    @property
    def training_raw(self) -> np.ndarray:
        return self.full_training[self.training_set]

    @property
    def forget_raw(self) -> np.ndarray:
        return self.full_training[self.training_set[self.forget]]

    @property
    def retain_raw(self) -> np.ndarray:
        return self.full_training[self.training_set[self.retrain]]

    @property
    def validation_raw(self) -> np.ndarray:
        return self.full_training[self.val]


@dataclass
class RunContext:
    """Resolved immutable inputs for one paired evaluation."""

    env_dir: Path
    config: dict[str, Any]
    repository_root: Path
    pretrained_checkpoint: Path
    unlearned_checkpoint: Path
    unlearned_run_dir: Path
    index_dir: Path
    partitions: dict[int, PartitionIndices]
    pretrained_sha256: str
    unlearned_sha256: str
    index_sha256: dict[str, str]
    unlearned_indices_verified: bool
    scenario: str
    method: str
    corruption_setting: str
    seed: int


@dataclass
class EvaluationResult:
    """In-memory and on-disk results for one environment."""

    context: RunContext
    summary: dict[str, Any]
    summary_table: pd.DataFrame
    per_sample: pd.DataFrame
    output_dir: Path
    output_files: dict[str, Path]


@dataclass(frozen=True)
class RuntimeConfig:
    """Optional Yeom settings read from the selected environment files."""

    unlearned_checkpoint: Optional[str]
    checkpoint_kind: str
    index_dir: Optional[str]
    output_dir: Optional[str]
    aggregate_output_dir: Optional[str]
    data_root: Optional[str]
    download: bool
    device: str
    batch_size: Optional[int]
    num_workers: int
    partition_seed: int
    forget_view: str
    backdoor_target: int
    scenario: Optional[str]
    method: Optional[str]
    utility_warning_accuracy_drop: float
    wandb_mode: str
    wandb_project: Optional[str]
    wandb_entity: Optional[str]
    wandb_group: Optional[str]
    wandb_run_name: Optional[str]
    wandb_tags: tuple[str, ...]


class IndexedEvaluationDataset(Dataset):
    """Dataset view that preserves raw indices and optional forget corruption."""

    def __init__(
        self,
        base_dataset: Dataset,
        references: Sequence[SampleRef],
        split: str,
        forget_view: str,
        corruption_setting: str,
        confuse_map: Mapping[int, int],
        backdoor_target: int,
        trigger_value: Optional[Sequence[float]],
    ) -> None:
        self.base_dataset = base_dataset
        self.references = list(references)
        self.split = split
        self.forget_view = forget_view
        self.corruption_setting = corruption_setting.upper()
        self.confuse_map = {int(key): int(value) for key, value in confuse_map.items()}
        self.backdoor_target = int(backdoor_target)
        self.trigger_value = trigger_value

    def __len__(self) -> int:
        return len(self.references)

    def __getitem__(self, index: int):
        reference = self.references[index]
        image, original_label = self.base_dataset[reference.source_index]
        original_label = _as_int(original_label)
        evaluation_label = original_label

        if self.split == SPLIT_FORGET and self.forget_view == "training":
            if self.corruption_setting == "CONFUSE":
                evaluation_label = self.confuse_map.get(original_label, original_label)
            elif self.corruption_setting == "BACKDOOR":
                image = _add_square_trigger(image, trigger_value=self.trigger_value)
                evaluation_label = self.backdoor_target

        return (
            image,
            evaluation_label,
            original_label,
            reference.source_index,
            reference.client_id,
        )


def _as_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_name(value: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-._")
    return cleaned or "yeom"


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(raw_path: str | Path, repository_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()

    candidates = [repository_root / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def _require_directory(path: Path, description: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{description} not found: {path}")


def _parse_client_id(path: Path) -> int:
    match = re.fullmatch(r"(\d+)_dataset_partitions\.npz", path.name)
    if match is None:
        raise ValueError(f"Unexpected partition-index filename: {path.name}")
    return int(match.group(1))


def _load_integer_vector(archive: Mapping[str, np.ndarray], key: str, path: Path) -> np.ndarray:
    value = np.asarray(archive[key])
    if value.ndim != 1:
        raise ValueError(f"{path}: index array {key!r} must be one-dimensional")
    if not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"{path}: index array {key!r} must contain integers")
    return value.astype(np.int64, copy=True)


def _validate_bounds(values: np.ndarray, upper_bound: int, name: str, path: Path) -> None:
    if values.size == 0:
        return
    if int(values.min()) < 0 or int(values.max()) >= upper_bound:
        raise ValueError(
            f"{path}: {name} contains an index outside [0, {upper_bound})"
        )


def _validate_unique(values: np.ndarray, name: str, path: Path) -> None:
    if len(np.unique(values)) != len(values):
        raise ValueError(f"{path}: {name} contains duplicate indices")


def _load_partition(path: Path) -> PartitionIndices:
    client_id = _parse_client_id(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = REQUIRED_INDEX_KEYS.difference(archive.files)
        if missing:
            raise ValueError(f"{path}: missing index arrays: {sorted(missing)}")
        arrays = {
            key: _load_integer_vector(archive, key, path)
            for key in REQUIRED_INDEX_KEYS
        }

    full_training = arrays["full_training"]
    training_set = arrays["training_set"]
    forget = arrays["forget"]
    retrain = arrays["retrain"]
    val = arrays["val"]
    test = arrays["test"]

    _validate_unique(full_training, "full_training", path)
    _validate_unique(training_set, "training_set", path)
    _validate_unique(forget, "forget", path)
    _validate_unique(retrain, "retrain", path)
    _validate_unique(val, "val", path)
    _validate_unique(test, "test", path)

    _validate_bounds(training_set, len(full_training), "training_set", path)
    _validate_bounds(val, len(full_training), "val", path)
    _validate_bounds(forget, len(training_set), "forget", path)
    _validate_bounds(retrain, len(training_set), "retrain", path)

    if np.intersect1d(training_set, val).size:
        raise ValueError(f"{path}: training_set and val are not disjoint")
    if np.intersect1d(forget, retrain).size:
        raise ValueError(f"{path}: forget and retrain are not disjoint")
    if len(forget) + len(retrain) != len(training_set):
        raise ValueError(
            f"{path}: forget and retrain do not cover the saved training_set"
        )

    return PartitionIndices(
        client_id=client_id,
        path=path,
        training_set=training_set,
        full_training=full_training,
        forget=forget,
        val=val,
        test=test,
        retrain=retrain,
    )


def _load_partitions(index_dir: Path, expected_clients: Optional[int]) -> dict[int, PartitionIndices]:
    _require_directory(index_dir, "partition-index directory")
    paths = sorted(index_dir.glob("*_dataset_partitions.npz"), key=_parse_client_id)
    if not paths:
        raise FileNotFoundError(f"No client partition files found in {index_dir}")

    partitions = {_parse_client_id(path): _load_partition(path) for path in paths}
    expected_ids = set(range(len(partitions)))
    if set(partitions) != expected_ids:
        raise ValueError(
            f"Partition IDs must be contiguous from zero; found {sorted(partitions)}"
        )
    if expected_clients is not None and len(partitions) != expected_clients:
        raise ValueError(
            f"Expected {expected_clients} partition files, found {len(partitions)} in {index_dir}"
        )

    _validate_cross_client_disjointness(partitions)
    return partitions


def _validate_cross_client_disjointness(partitions: Mapping[int, PartitionIndices]) -> None:
    seen_training: dict[int, int] = {}
    seen_test: dict[int, int] = {}
    for client_id, partition in partitions.items():
        for raw_index in partition.full_training.tolist():
            previous = seen_training.setdefault(int(raw_index), client_id)
            if previous != client_id:
                raise ValueError(
                    f"Training index {raw_index} occurs in clients {previous} and {client_id}"
                )
        for raw_index in partition.test.tolist():
            previous = seen_test.setdefault(int(raw_index), client_id)
            if previous != client_id:
                raise ValueError(
                    f"Test index {raw_index} occurs in clients {previous} and {client_id}"
                )


def _compare_index_directories(reference: Path, candidate: Path) -> None:
    reference_files = {path.name: path for path in reference.glob("*_dataset_partitions.npz")}
    candidate_files = {path.name: path for path in candidate.glob("*_dataset_partitions.npz")}
    if set(reference_files) != set(candidate_files):
        raise ValueError(
            "Pretrained and unlearned partition directories contain different client files"
        )

    for name in sorted(reference_files):
        with np.load(reference_files[name], allow_pickle=False) as expected, np.load(
            candidate_files[name], allow_pickle=False
        ) as actual:
            if set(expected.files) != set(actual.files):
                raise ValueError(f"Partition schemas differ for {name}")
            for key in expected.files:
                if not np.array_equal(expected[key], actual[key]):
                    raise ValueError(
                        f"Unlearned partition indices differ from pretrained indices: {name}:{key}"
                    )


def _infer_scenario(env_dir: Path, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    lowered = {part.lower() for part in env_dir.parts}
    if "client_level" in lowered:
        return "client-level"
    if "data_level" in lowered:
        return "data-level"
    return "unspecified"


def _infer_method(env_dir: Path, config: Mapping[str, Any], explicit: Optional[str]) -> str:
    if explicit:
        return explicit

    parts = list(env_dir.parts)
    lower_parts = [part.lower() for part in parts]
    for corruption in ("privacy", "confuse", "backdoor"):
        if corruption not in lower_parts:
            continue
        index = lower_parts.index(corruption) + 1
        if index >= len(parts):
            break
        method = parts[index]
        if method.lower() == "unlearn" and index + 1 < len(parts):
            variant = parts[index + 1]
            if not re.fullmatch(r"N_\d+|\d+", variant, flags=re.IGNORECASE):
                method = f"{method}-{variant}"
        return method

    return str(config.get("NAME", config.get("STARTING_PHASE", "unknown")))


def _expected_client_count(config: Mapping[str, Any]) -> Optional[int]:
    raw_value = config.get("NUM_SUPERNODES", config.get("MIN_CLIENTS"))
    if raw_value in (None, ""):
        return None
    return int(raw_value)


def _resolve_run_context(
    settings: RuntimeConfig, env_dir_value: str | Path
) -> RunContext:
    repository_root = _repository_root()
    env_dir = _resolve_path(env_dir_value, repository_root)
    _require_directory(env_dir, "unlearning environment directory")

    config = load_config(str(env_dir))
    resume = str(config.get("RESUME", "")).strip()
    if not resume or resume.lower() in {"none", "null"}:
        raise ValueError(f"Unlearning environment has no pretrained RESUME checkpoint: {env_dir}")

    pretrained_checkpoint = _resolve_path(resume, repository_root)
    _require_file(pretrained_checkpoint, "pretrained checkpoint from RESUME")

    if settings.unlearned_checkpoint:
        unlearned_checkpoint = _resolve_path(
            settings.unlearned_checkpoint, repository_root
        )
        unlearned_run_dir = unlearned_checkpoint.parent.parent
    else:
        generated_run_dir = _resolve_path(generate_save_path(config), repository_root)
        checkpoint_name = (
            "model_best.pth"
            if settings.checkpoint_kind == "best"
            else "model_latest.pth"
        )
        unlearned_checkpoint = generated_run_dir / "models_chkpts" / checkpoint_name
        unlearned_run_dir = generated_run_dir
    _require_file(unlearned_checkpoint, "unlearned checkpoint")

    if settings.index_dir:
        index_dir = _resolve_path(settings.index_dir, repository_root)
    else:
        index_dir = pretrained_checkpoint.parent.parent / "partition_indexes"

    partitions = _load_partitions(index_dir, _expected_client_count(config))

    unlearned_index_dir = unlearned_run_dir / "partition_indexes"
    indices_verified = False
    if unlearned_index_dir.is_dir():
        _compare_index_directories(index_dir, unlearned_index_dir)
        indices_verified = True

    index_hashes = {
        partition.path.name: _sha256_file(partition.path)
        for partition in partitions.values()
    }
    return RunContext(
        env_dir=env_dir,
        config=config,
        repository_root=repository_root,
        pretrained_checkpoint=pretrained_checkpoint,
        unlearned_checkpoint=unlearned_checkpoint,
        unlearned_run_dir=unlearned_run_dir,
        index_dir=index_dir,
        partitions=partitions,
        pretrained_sha256=_sha256_file(pretrained_checkpoint),
        unlearned_sha256=_sha256_file(unlearned_checkpoint),
        index_sha256=index_hashes,
        unlearned_indices_verified=indices_verified,
        scenario=_infer_scenario(env_dir, settings.scenario),
        method=_infer_method(env_dir, config, settings.method),
        corruption_setting=str(config.get("UNLEARNING_CASE", "PRIVACY")).upper(),
        seed=int(config.get("SEED", 0)),
    )


def _build_sample_references(
    context: RunContext,
) -> tuple[list[SampleRef], list[SampleRef], list[SampleRef]]:
    affected_clients = {int(value) for value in context.config.get("CLIENT_ID_TO_FORGET", [])}
    if not affected_clients:
        raise ValueError("CLIENT_ID_TO_FORGET is empty; the forget set is undefined")
    missing_clients = affected_clients.difference(context.partitions)
    if missing_clients:
        raise ValueError(
            f"Forget-client IDs have no saved partition file: {sorted(missing_clients)}"
        )

    forget_keys: set[tuple[int, int]] = set()
    forget_refs: list[SampleRef] = []
    all_training_refs: list[SampleRef] = []
    nonmember_refs: list[SampleRef] = []

    for client_id, partition in sorted(context.partitions.items()):
        for raw_index in partition.training_raw.tolist():
            all_training_refs.append(SampleRef(client_id, int(raw_index)))
        if client_id in affected_clients:
            for raw_index in partition.forget_raw.tolist():
                reference = SampleRef(client_id, int(raw_index))
                forget_refs.append(reference)
                forget_keys.add((reference.client_id, reference.source_index))
        for raw_index in partition.test.tolist():
            nonmember_refs.append(SampleRef(client_id, int(raw_index)))

    retain_refs = [
        reference
        for reference in all_training_refs
        if (reference.client_id, reference.source_index) not in forget_keys
    ]

    forget_refs.sort(key=lambda value: (value.client_id, value.source_index))
    retain_refs.sort(key=lambda value: (value.client_id, value.source_index))
    nonmember_refs.sort(key=lambda value: (value.client_id, value.source_index))

    if not forget_refs:
        raise ValueError("Resolved forget set is empty")
    if not retain_refs:
        raise ValueError("Resolved retained set is empty")
    if not nonmember_refs:
        raise ValueError("Resolved non-member set is empty")

    if len(forget_refs) + len(retain_refs) != len(all_training_refs):
        raise ValueError("D_f and D_r do not reconstruct the original training set")

    return forget_refs, retain_refs, nonmember_refs


def _build_base_datasets(
    context: RunContext,
    data_root: Path,
    download: bool,
    partition_seed: int,
) -> tuple[Dataset, Dataset]:
    dataset_name = str(context.config["DATASET"]).lower().replace("-", "").replace("_", "")
    root = str(data_root)

    if dataset_name == "cifar100":
        evaluation_transform = _cifar100_eval_transform()
        return (
            datasets.CIFAR100(root=root, train=True, download=download, transform=evaluation_transform),
            datasets.CIFAR100(root=root, train=False, download=download, transform=evaluation_transform),
        )
    if dataset_name == "cifar10":
        evaluation_transform = transforms.ToTensor()
        return (
            datasets.CIFAR10(root=root, train=True, download=download, transform=evaluation_transform),
            datasets.CIFAR10(root=root, train=False, download=download, transform=evaluation_transform),
        )
    if dataset_name == "mnist":
        evaluation_transform = transforms.ToTensor()
        return (
            datasets.MNIST(root=root, train=True, download=download, transform=evaluation_transform),
            datasets.MNIST(root=root, train=False, download=download, transform=evaluation_transform),
        )
    if dataset_name in {"fashionmnist", "fashmnist"}:
        evaluation_transform = transforms.ToTensor()
        return (
            datasets.FashionMNIST(
                root=root, train=True, download=download, transform=evaluation_transform
            ),
            datasets.FashionMNIST(
                root=root, train=False, download=download, transform=evaluation_transform
            ),
        )
    if dataset_name == "caltech101":
        evaluation_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        return (
            Caltech101Wrapper(
                root=root,
                train=True,
                download=download,
                transform=evaluation_transform,
                seed=partition_seed,
            ),
            Caltech101Wrapper(
                root=root,
                train=False,
                download=download,
                transform=evaluation_transform,
                seed=partition_seed,
            ),
        )

    raise ValueError(f"Unsupported dataset for Yeom evaluation: {context.config['DATASET']}")


def _validate_source_ranges(
    train_dataset: Dataset,
    test_dataset: Dataset,
    forget_refs: Sequence[SampleRef],
    retain_refs: Sequence[SampleRef],
    nonmember_refs: Sequence[SampleRef],
) -> None:
    train_indices = [reference.source_index for reference in (*forget_refs, *retain_refs)]
    test_indices = [reference.source_index for reference in nonmember_refs]
    if min(train_indices) < 0 or max(train_indices) >= len(train_dataset):
        raise ValueError(
            f"Saved training indices are incompatible with dataset length {len(train_dataset)}"
        )
    if min(test_indices) < 0 or max(test_indices) >= len(test_dataset):
        raise ValueError(
            f"Saved test indices are incompatible with dataset length {len(test_dataset)}"
        )


def _label_metadata(
    split: str,
    forget_view: str,
    corruption_setting: str,
) -> tuple[str, str]:
    if split != SPLIT_FORGET or forget_view == "clean":
        return "clean_ground_truth", "clean"
    if corruption_setting == "CONFUSE":
        return "confused_training_label", "clean"
    if corruption_setting == "BACKDOOR":
        return "backdoor_target_label", "backdoored"
    return "clean_ground_truth", "clean"


def _build_evaluation_datasets(
    context: RunContext,
    settings: RuntimeConfig,
) -> dict[str, IndexedEvaluationDataset]:
    forget_refs, retain_refs, nonmember_refs = _build_sample_references(context)
    data_root = (
        _resolve_path(settings.data_root, context.repository_root)
        if settings.data_root
        else context.repository_root / "data"
    )
    train_dataset, test_dataset = _build_base_datasets(
        context,
        data_root=data_root,
        download=settings.download,
        partition_seed=settings.partition_seed,
    )
    _validate_source_ranges(
        train_dataset,
        test_dataset,
        forget_refs,
        retain_refs,
        nonmember_refs,
    )

    confuse_map = context.config.get("MAP_CONFUSE", {}) or {}
    if (
        settings.forget_view == "training"
        and context.corruption_setting == "CONFUSE"
        and not confuse_map
    ):
        raise ValueError("Training-view confusion evaluation requires MAP_CONFUSE")

    dataset_name = str(context.config["DATASET"]).lower()
    trigger_value = (
        CIFAR100_NORMALIZED_WHITE if dataset_name == "cifar100" else None
    )
    common = {
        "forget_view": settings.forget_view,
        "corruption_setting": context.corruption_setting,
        "confuse_map": confuse_map,
        "backdoor_target": settings.backdoor_target,
        "trigger_value": trigger_value,
    }
    return {
        SPLIT_FORGET: IndexedEvaluationDataset(
            train_dataset, forget_refs, SPLIT_FORGET, **common
        ),
        SPLIT_RETAIN: IndexedEvaluationDataset(
            train_dataset, retain_refs, SPLIT_RETAIN, **common
        ),
        SPLIT_NONMEMBER: IndexedEvaluationDataset(
            test_dataset, nonmember_refs, SPLIT_NONMEMBER, **common
        ),
    }


def _resolve_device(requested: str, config: Mapping[str, Any]) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested}")
        return device

    configured = str(config.get("DEVICE", "cuda:0"))
    if torch.cuda.is_available():
        return torch.device(configured)
    return torch.device("cpu")


def _set_deterministic_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _extract_logits(outputs: Any) -> torch.Tensor:
    if hasattr(outputs, "logits"):
        outputs = outputs.logits
    elif isinstance(outputs, (tuple, list)):
        outputs = outputs[0]
    if not isinstance(outputs, torch.Tensor) or outputs.ndim != 2:
        raise ValueError("Model output must be a two-dimensional logits tensor")
    return outputs


def _infer_split(
    model: torch.nn.Module,
    loader: DataLoader,
    split: str,
    model_label: str,
    checkpoint_hash: str,
    context: RunContext,
    device: torch.device,
    forget_view: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    label_basis, input_variant = _label_metadata(
        split, forget_view, context.corruption_setting
    )
    num_classes = int(context.config["NUM_CLASSES"])

    with torch.inference_mode():
        for batch in loader:
            images, labels, original_labels, sample_indices, client_ids = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            outputs = _extract_logits(model(images))
            if outputs.shape[1] != num_classes:
                raise ValueError(
                    f"Model produced {outputs.shape[1]} classes, expected {num_classes}"
                )

            losses = F.cross_entropy(outputs, labels, reduction="none")
            probabilities = F.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)
            true_probabilities = probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)

            labels_cpu = labels.cpu().tolist()
            original_cpu = original_labels.cpu().tolist()
            predictions_cpu = predictions.cpu().tolist()
            probabilities_cpu = true_probabilities.cpu().tolist()
            losses_cpu = losses.cpu().tolist()
            indices_cpu = sample_indices.cpu().tolist()
            client_ids_cpu = client_ids.cpu().tolist()

            for position in range(len(labels_cpu)):
                raw_index = int(indices_cpu[position])
                client_id = int(client_ids_cpu[position])
                evaluation_label = int(labels_cpu[position])
                prediction = int(predictions_cpu[position])
                loss_value = float(losses_cpu[position])
                source_dataset = "test" if split == SPLIT_NONMEMBER else "train"
                rows.append(
                    {
                        "experiment_seed": context.seed,
                        "scenario": context.scenario,
                        "model": model_label,
                        "checkpoint_sha256": checkpoint_hash,
                        "dataset": str(context.config["DATASET"]),
                        "dataset_split": split,
                        "source_dataset": source_dataset,
                        "stable_sample_index": raw_index,
                        "stable_sample_id": f"{source_dataset}:{raw_index}",
                        "client_id": client_id,
                        "true_label": evaluation_label,
                        "original_label": int(original_cpu[position]),
                        "predicted_label": prediction,
                        "correct": int(prediction == evaluation_label),
                        "true_class_probability": float(probabilities_cpu[position]),
                        "cross_entropy_loss": loss_value,
                        "membership_score": -loss_value,
                        "yeom_threshold": np.nan,
                        "yeom_member": 0,
                        "unlearning_method": context.method,
                        "corruption_setting": context.corruption_setting,
                        "label_basis": label_basis,
                        "input_variant": input_variant,
                    }
                )

    if not rows:
        raise ValueError(f"No samples were evaluated for split {split}")
    return pd.DataFrame(rows)


def _evaluate_model(
    context: RunContext,
    datasets_by_split: Mapping[str, Dataset],
    checkpoint: Path,
    checkpoint_hash: str,
    model_label: str,
    settings: RuntimeConfig,
    device: torch.device,
) -> pd.DataFrame:
    model = load_model(
        str(context.config["MODEL"]),
        str(checkpoint),
        strict_checkpoint=True,
    )
    model.to(device)
    model.eval()

    batch_size = settings.batch_size or int(context.config.get("VAL_BATCH", 256))
    frames = []
    for split in SPLIT_ORDER:
        loader = DataLoader(
            datasets_by_split[split],
            batch_size=batch_size,
            shuffle=False,
            num_workers=settings.num_workers,
            pin_memory=device.type == "cuda",
        )
        frames.append(
            _infer_split(
                model=model,
                loader=loader,
                split=split,
                model_label=model_label,
                checkpoint_hash=checkpoint_hash,
                context=context,
                device=device,
                forget_view=settings.forget_view,
            )
        )

    model.to("cpu")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pd.concat(frames, ignore_index=True)


def _apply_canonical_thresholds(per_sample: pd.DataFrame) -> dict[str, float]:
    pretrained_training = per_sample[
        (per_sample["model"] == MODEL_PRETRAINED)
        & (per_sample["dataset_split"].isin([SPLIT_FORGET, SPLIT_RETAIN]))
    ]
    unlearned_retain = per_sample[
        (per_sample["model"] == MODEL_UNLEARNED)
        & (per_sample["dataset_split"] == SPLIT_RETAIN)
    ]
    if pretrained_training.empty or unlearned_retain.empty:
        raise ValueError("Cannot compute canonical Yeom thresholds from empty training sets")

    thresholds = {
        MODEL_PRETRAINED: float(pretrained_training["cross_entropy_loss"].mean()),
        MODEL_UNLEARNED: float(unlearned_retain["cross_entropy_loss"].mean()),
    }
    for model_label, threshold in thresholds.items():
        mask = per_sample["model"] == model_label
        per_sample.loc[mask, "yeom_threshold"] = threshold
        per_sample.loc[mask, "yeom_member"] = (
            per_sample.loc[mask, "cross_entropy_loss"] <= threshold
        ).astype(int)
    return thresholds


def _descriptive_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    losses = frame["cross_entropy_loss"].to_numpy(dtype=float)
    scores = frame["membership_score"].to_numpy(dtype=float)
    return {
        "n": int(len(frame)),
        "loss_mean": float(np.mean(losses)),
        "loss_std": float(np.std(losses, ddof=0)),
        "loss_median": float(np.median(losses)),
        "loss_min": float(np.min(losses)),
        "loss_max": float(np.max(losses)),
        "loss_q05": float(np.quantile(losses, 0.05)),
        "loss_q25": float(np.quantile(losses, 0.25)),
        "loss_q75": float(np.quantile(losses, 0.75)),
        "loss_q95": float(np.quantile(losses, 0.95)),
        "membership_score_mean": float(np.mean(scores)),
        "accuracy": float(frame["correct"].mean()),
        "accuracy_pct": float(frame["correct"].mean() * 100.0),
        "predicted_member_rate": float(frame["yeom_member"].mean()),
        "predicted_member_rate_pct": float(frame["yeom_member"].mean() * 100.0),
    }


def _tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, target_fpr: float) -> float:
    eligible = tpr[fpr <= target_fpr + np.finfo(float).eps]
    return float(np.max(eligible)) if eligible.size else 0.0


def _loss_attack_metrics(model_frame: pd.DataFrame) -> dict[str, Any]:
    forget = model_frame[model_frame["dataset_split"] == SPLIT_FORGET]
    nonmember = model_frame[model_frame["dataset_split"] == SPLIT_NONMEMBER]
    labels = np.concatenate(
        [np.ones(len(forget), dtype=int), np.zeros(len(nonmember), dtype=int)]
    )
    scores = np.concatenate(
        [
            forget["membership_score"].to_numpy(dtype=float),
            nonmember["membership_score"].to_numpy(dtype=float),
        ]
    )
    auc = float(roc_auc_score(labels, scores))
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    advantage = float(np.max(np.abs(tpr - fpr)))

    metrics: dict[str, Any] = {
        "forget_vs_nonmember_auc": auc,
        "attack_advantage": advantage,
        "tpr_at_1pct_fpr": _tpr_at_fpr(fpr, tpr, 0.01),
        "tpr_at_0_1pct_fpr": None,
        "tpr_at_0_1pct_fpr_omission_reason": None,
    }
    if len(nonmember) >= 1000:
        metrics["tpr_at_0_1pct_fpr"] = _tpr_at_fpr(fpr, tpr, 0.001)
    else:
        metrics["tpr_at_0_1pct_fpr_omission_reason"] = (
            f"Only {len(nonmember)} non-members; at least 1000 are required "
            "for 0.1% FPR resolution."
        )
    return metrics


def _model_metrics(
    per_sample: pd.DataFrame,
    model_label: str,
    threshold: float,
) -> dict[str, Any]:
    model_frame = per_sample[per_sample["model"] == model_label]
    split_metrics = {
        split: _descriptive_metrics(
            model_frame[model_frame["dataset_split"] == split]
        )
        for split in SPLIT_ORDER
    }
    return {
        "yeom_threshold": threshold,
        "threshold_source": (
            "original_training_set_Dr_union_Df"
            if model_label == MODEL_PRETRAINED
            else "retained_set_Dr_effective_training_set"
        ),
        "splits": split_metrics,
        "loss_attack": _loss_attack_metrics(model_frame),
    }


def _comparison_metrics(
    pretrained: Mapping[str, Any],
    unlearned: Mapping[str, Any],
    utility_warning_accuracy_drop: float,
) -> dict[str, Any]:
    pre_forget = pretrained["splits"][SPLIT_FORGET]["predicted_member_rate"]
    unl_forget = unlearned["splits"][SPLIT_FORGET]["predicted_member_rate"]
    pre_auc = pretrained["loss_attack"]["forget_vs_nonmember_auc"]
    unl_auc = unlearned["loss_attack"]["forget_vs_nonmember_auc"]

    retain_accuracy_drop = (
        pretrained["splits"][SPLIT_RETAIN]["accuracy"]
        - unlearned["splits"][SPLIT_RETAIN]["accuracy"]
    )
    nonmember_accuracy_drop = (
        pretrained["splits"][SPLIT_NONMEMBER]["accuracy"]
        - unlearned["splits"][SPLIT_NONMEMBER]["accuracy"]
    )
    utility_warning = (
        retain_accuracy_drop >= utility_warning_accuracy_drop
        or nonmember_accuracy_drop >= utility_warning_accuracy_drop
    )

    return {
        "delta_forget_member_rate": float(unl_forget - pre_forget),
        "delta_forget_member_rate_pct_points": float((unl_forget - pre_forget) * 100.0),
        "forget_member_rate_reduction": float(pre_forget - unl_forget),
        "forget_member_rate_reduction_pct_points": float((pre_forget - unl_forget) * 100.0),
        "auc_distance_to_chance_change": float(abs(unl_auc - 0.5) - abs(pre_auc - 0.5)),
        "retain_accuracy_change": float(
            unlearned["splits"][SPLIT_RETAIN]["accuracy"]
            - pretrained["splits"][SPLIT_RETAIN]["accuracy"]
        ),
        "retain_accuracy_change_pct_points": float(-retain_accuracy_drop * 100.0),
        "nonmember_accuracy_change": float(
            unlearned["splits"][SPLIT_NONMEMBER]["accuracy"]
            - pretrained["splits"][SPLIT_NONMEMBER]["accuracy"]
        ),
        "nonmember_accuracy_change_pct_points": float(-nonmember_accuracy_drop * 100.0),
        "forget_accuracy_change": float(
            unlearned["splits"][SPLIT_FORGET]["accuracy"]
            - pretrained["splits"][SPLIT_FORGET]["accuracy"]
        ),
        "retain_loss_change": float(
            unlearned["splits"][SPLIT_RETAIN]["loss_mean"]
            - pretrained["splits"][SPLIT_RETAIN]["loss_mean"]
        ),
        "nonmember_loss_change": float(
            unlearned["splits"][SPLIT_NONMEMBER]["loss_mean"]
            - pretrained["splits"][SPLIT_NONMEMBER]["loss_mean"]
        ),
        "utility_damage_warning": bool(utility_warning),
        "utility_warning_accuracy_drop_threshold": float(utility_warning_accuracy_drop),
        "utility_warning_reason": (
            "Retained or non-member accuracy dropped by at least "
            f"{utility_warning_accuracy_drop * 100.0:.2f} percentage points."
            if utility_warning
            else None
        ),
    }


def _summary_table(
    metrics_by_model: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    rows = []
    for model_label in MODEL_ORDER:
        metrics = metrics_by_model[model_label]
        rows.append(
            {
                "model": model_label,
                "yeom_threshold": metrics["yeom_threshold"],
                "forget_member_rate_pct": metrics["splits"][SPLIT_FORGET][
                    "predicted_member_rate_pct"
                ],
                "retain_member_rate_pct": metrics["splits"][SPLIT_RETAIN][
                    "predicted_member_rate_pct"
                ],
                "nonmember_fpr_pct": metrics["splits"][SPLIT_NONMEMBER][
                    "predicted_member_rate_pct"
                ],
                "forget_vs_nonmember_auc": metrics["loss_attack"][
                    "forget_vs_nonmember_auc"
                ],
                "attack_advantage": metrics["loss_attack"]["attack_advantage"],
                "tpr_at_1pct_fpr": metrics["loss_attack"]["tpr_at_1pct_fpr"],
                "tpr_at_0_1pct_fpr": metrics["loss_attack"]["tpr_at_0_1pct_fpr"],
                "forget_accuracy_pct": metrics["splits"][SPLIT_FORGET]["accuracy_pct"],
                "retain_accuracy_pct": metrics["splits"][SPLIT_RETAIN]["accuracy_pct"],
                "test_accuracy_pct": metrics["splits"][SPLIT_NONMEMBER]["accuracy_pct"],
                "forget_loss": metrics["splits"][SPLIT_FORGET]["loss_mean"],
                "retain_loss": metrics["splits"][SPLIT_RETAIN]["loss_mean"],
                "test_loss": metrics["splits"][SPLIT_NONMEMBER]["loss_mean"],
            }
        )
    return pd.DataFrame(rows)


def _selected_config_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    keys = [
        "NAME",
        "MODEL",
        "DATASET",
        "NUM_CLASSES",
        "SEED",
        "CONFIG_ID",
        "CONFIG_NUMBER",
        "STARTING_PHASE",
        "CLIENT_ID_TO_FORGET",
        "Client_ID_TO_EXIT",
        "FORGET_CLASS",
        "UNLEARNING_CASE",
        "MAP_CONFUSE",
        "NON_IID_DP",
        "NUM_SUPERNODES",
        "MIN_CLIENTS",
        "LOSSCLS",
        "LOSSDIV",
        "LOSSKD",
    ]
    return {key: config[key] for key in keys if key in config}


def _build_summary_payload(
    context: RunContext,
    settings: RuntimeConfig,
    metrics_by_model: Mapping[str, Mapping[str, Any]],
    comparison: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    label_basis, input_variant = _label_metadata(
        SPLIT_FORGET, settings.forget_view, context.corruption_setting
    )
    return {
        "evaluation": "canonical_yeom_federated_unlearning",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment_directory": str(context.env_dir),
        "scenario": context.scenario,
        "unlearning_method": context.method,
        "corruption_setting": context.corruption_setting,
        "forget_view": settings.forget_view,
        "forget_label_basis": label_basis,
        "forget_input_variant": input_variant,
        "device": str(device),
        "batch_size": int(
            settings.batch_size or context.config.get("VAL_BATCH", 256)
        ),
        "seed": context.seed,
        "config": _selected_config_metadata(context.config),
        "checkpoints": {
            MODEL_PRETRAINED: {
                "path": str(context.pretrained_checkpoint),
                "sha256": context.pretrained_sha256,
            },
            MODEL_UNLEARNED: {
                "path": str(context.unlearned_checkpoint),
                "sha256": context.unlearned_sha256,
            },
        },
        "indices": {
            "canonical_source": str(context.index_dir),
            "sha256_by_file": context.index_sha256,
            "unlearned_indices_present_and_identical": context.unlearned_indices_verified,
            "raw_index_resolution": "full_training[training_set[local_split_index]]",
        },
        "set_definition": {
            "forget": "saved forget subsets for CLIENT_ID_TO_FORGET",
            "retain": "original training set minus forget set",
            "non_member": "union of saved official test-set partitions",
        },
        "models": dict(metrics_by_model),
        "comparison": dict(comparison),
    }


def _resolve_output_dir(
    context: RunContext,
    settings: RuntimeConfig,
    multiple_runs: bool,
) -> Path:
    if not settings.output_dir:
        return (
            context.unlearned_run_dir / "attacks" / "yeom" / settings.forget_view
        )

    base = _resolve_path(settings.output_dir, context.repository_root)
    if not multiple_runs:
        return base
    suffix = _safe_name(
        f"{context.config['DATASET']}-{context.scenario}-{context.method}-seed{context.seed}-"
        f"{context.config.get('CONFIG_NUMBER', 'run')}"
    )
    return base / suffix


def _write_result_files(
    output_dir: Path,
    per_sample: pd.DataFrame,
    summary_table: pd.DataFrame,
    summary: Mapping[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "per_sample": output_dir / "per_sample.csv",
        "summary_table": output_dir / "summary.csv",
        "summary": output_dir / "summary.json",
    }
    per_sample.to_csv(files["per_sample"], index=False, float_format="%.12g")
    summary_table.to_csv(files["summary_table"], index=False, float_format="%.12g")
    _write_json(files["summary"], summary)
    return files


def _wandb_scalar_metrics(summary: Mapping[str, Any]) -> dict[str, float | int]:
    logged: dict[str, float | int] = {}
    for model_label, prefix in (
        (MODEL_PRETRAINED, "yeom/pre"),
        (MODEL_UNLEARNED, "yeom/unlearn"),
    ):
        metrics = summary["models"][model_label]
        logged[f"{prefix}/threshold"] = metrics["yeom_threshold"]
        for split in SPLIT_ORDER:
            safe_split = split.replace("-", "_")
            split_metrics = metrics["splits"][split]
            for key in (
                "n",
                "loss_mean",
                "loss_std",
                "loss_median",
                "loss_min",
                "loss_max",
                "loss_q05",
                "loss_q25",
                "loss_q75",
                "loss_q95",
                "membership_score_mean",
                "accuracy",
                "accuracy_pct",
                "predicted_member_rate",
                "predicted_member_rate_pct",
            ):
                logged[f"{prefix}/{safe_split}/{key}"] = split_metrics[key]
        for key, value in metrics["loss_attack"].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                logged[f"{prefix}/loss_attack/{key}"] = value

    for key, value in summary["comparison"].items():
        if isinstance(value, bool):
            logged[f"yeom/comparison/{key}"] = int(value)
        elif isinstance(value, (int, float)) and value is not None:
            logged[f"yeom/comparison/{key}"] = value
    return logged


def _wandb_config(context: RunContext, settings: RuntimeConfig) -> dict[str, Any]:
    config = _selected_config_metadata(context.config)
    config.update(
        {
            "yeom_scenario": context.scenario,
            "yeom_method": context.method,
            "yeom_forget_view": settings.forget_view,
            "yeom_pretrained_checkpoint": str(context.pretrained_checkpoint),
            "yeom_unlearned_checkpoint": str(context.unlearned_checkpoint),
            "yeom_pretrained_sha256": context.pretrained_sha256,
            "yeom_unlearned_sha256": context.unlearned_sha256,
            "yeom_index_dir": str(context.index_dir),
        }
    )
    return json.loads(json.dumps(config, default=_json_default))


def _log_result_to_wandb(
    result: EvaluationResult,
    settings: RuntimeConfig,
    group: str,
) -> None:
    if settings.wandb_mode == "disabled":
        return
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "W&B logging was requested but wandb is not installed. Local outputs were written."
        ) from error

    context = result.context
    project = (
        settings.wandb_project
        or context.config.get("WANDB_PROJECT")
        or context.config.get("NAME", "fed-scrub")
    )
    run_name = settings.wandb_run_name or _safe_name(
        f"yeom-{context.config['DATASET']}-{context.method}-seed{context.seed}-"
        f"{context.unlearned_sha256[:8]}"
    )
    tags = ["yeom", "membership-inference", "federated-unlearning"] + list(
        settings.wandb_tags or []
    )
    run = None
    try:
        run = wandb.init(
            project=str(project),
            entity=settings.wandb_entity,
            name=run_name,
            group=group,
            job_type="yeom-evaluation",
            tags=tags,
            mode=settings.wandb_mode,
            config=_wandb_config(context, settings),
            reinit=True,
        )
        run.log(_wandb_scalar_metrics(result.summary))
        run.log({"yeom/summary_table": wandb.Table(dataframe=result.summary_table)})

        artifact_name = _safe_name(
            f"yeom-{context.config['DATASET']}-{context.method}-seed{context.seed}-"
            f"{context.unlearned_sha256[:8]}"
        )
        artifact = wandb.Artifact(
            name=artifact_name,
            type="yeom-evaluation",
            metadata={
                "seed": context.seed,
                "scenario": context.scenario,
                "method": context.method,
                "pretrained_sha256": context.pretrained_sha256,
                "unlearned_sha256": context.unlearned_sha256,
            },
        )
        for path in result.output_files.values():
            artifact.add_file(str(path))
        run.log_artifact(artifact)
    except Exception as error:
        raise RuntimeError(
            "W&B upload failed after local Yeom outputs were written to "
            f"{result.output_dir}: {error}"
        ) from error
    finally:
        if run is not None:
            run.finish()


def evaluate_environment(
    settings: RuntimeConfig,
    env_dir: str | Path,
    multiple_runs: bool = False,
) -> EvaluationResult:
    """Evaluate one unlearning environment and write its local outputs."""

    context = _resolve_run_context(settings, env_dir)
    _set_deterministic_seed(context.seed)
    datasets_by_split = _build_evaluation_datasets(context, settings)
    device = _resolve_device(settings.device, context.config)

    pretrained_frame = _evaluate_model(
        context=context,
        datasets_by_split=datasets_by_split,
        checkpoint=context.pretrained_checkpoint,
        checkpoint_hash=context.pretrained_sha256,
        model_label=MODEL_PRETRAINED,
        settings=settings,
        device=device,
    )
    unlearned_frame = _evaluate_model(
        context=context,
        datasets_by_split=datasets_by_split,
        checkpoint=context.unlearned_checkpoint,
        checkpoint_hash=context.unlearned_sha256,
        model_label=MODEL_UNLEARNED,
        settings=settings,
        device=device,
    )
    per_sample = pd.concat([pretrained_frame, unlearned_frame], ignore_index=True)
    thresholds = _apply_canonical_thresholds(per_sample)
    metrics_by_model = {
        model_label: _model_metrics(per_sample, model_label, thresholds[model_label])
        for model_label in MODEL_ORDER
    }
    comparison = _comparison_metrics(
        metrics_by_model[MODEL_PRETRAINED],
        metrics_by_model[MODEL_UNLEARNED],
        utility_warning_accuracy_drop=settings.utility_warning_accuracy_drop,
    )
    summary_table = _summary_table(metrics_by_model)
    summary = _build_summary_payload(
        context=context,
        settings=settings,
        metrics_by_model=metrics_by_model,
        comparison=comparison,
        device=device,
    )

    model_rank = {name: rank for rank, name in enumerate(MODEL_ORDER)}
    split_rank = {name: rank for rank, name in enumerate(SPLIT_ORDER)}
    per_sample["_model_rank"] = per_sample["model"].map(model_rank)
    per_sample["_split_rank"] = per_sample["dataset_split"].map(split_rank)
    per_sample.sort_values(
        ["_model_rank", "_split_rank", "client_id", "stable_sample_index"],
        inplace=True,
    )
    per_sample.drop(columns=["_model_rank", "_split_rank"], inplace=True)
    per_sample.reset_index(drop=True, inplace=True)

    output_dir = _resolve_output_dir(context, settings, multiple_runs)
    output_files = _write_result_files(output_dir, per_sample, summary_table, summary)
    return EvaluationResult(
        context=context,
        summary=summary,
        summary_table=summary_table,
        per_sample=per_sample,
        output_dir=output_dir,
        output_files=output_files,
    )


def _numeric_columns(frame: pd.DataFrame, excluded: Iterable[str]) -> list[str]:
    excluded_set = set(excluded)
    return [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in excluded_set
    ]


def _aggregate_rows(
    frame: pd.DataFrame,
    section: str,
    excluded: Iterable[str] = (),
) -> list[dict[str, Any]]:
    rows = []
    for column in _numeric_columns(frame, excluded):
        values = frame[column].dropna().to_numpy(dtype=float)
        if not len(values):
            continue
        rows.append(
            {
                "section": section,
                "metric": column,
                "n": int(len(values)),
                "mean": float(np.mean(values)),
                "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            }
        )
    return rows


def _aggregate_results(
    results: Sequence[EvaluationResult],
    settings: RuntimeConfig,
) -> tuple[Path, dict[str, Path], pd.DataFrame]:
    repository_root = results[0].context.repository_root
    if settings.aggregate_output_dir:
        output_dir = _resolve_path(
            settings.aggregate_output_dir, repository_root
        )
    else:
        output_dir = (
            repository_root
            / "checkpoints"
            / "attacks"
            / "yeom"
            / f"aggregate-{_utc_timestamp()}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows = []
    comparison_rows = []
    for result in results:
        for _, row in result.summary_table.iterrows():
            record = row.to_dict()
            record.update(
                {
                    "seed": result.context.seed,
                    "environment_directory": str(result.context.env_dir),
                    "pretrained_sha256": result.context.pretrained_sha256,
                    "unlearned_sha256": result.context.unlearned_sha256,
                }
            )
            per_seed_rows.append(record)
        comparison_record = {
            key: value
            for key, value in result.summary["comparison"].items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        comparison_record.update(
            {
                "seed": result.context.seed,
                "environment_directory": str(result.context.env_dir),
            }
        )
        comparison_rows.append(comparison_record)

    per_seed = pd.DataFrame(per_seed_rows)
    comparisons = pd.DataFrame(comparison_rows)
    unlearned = per_seed[per_seed["model"] == MODEL_UNLEARNED]

    pretrained = per_seed[per_seed["model"] == MODEL_PRETRAINED].drop_duplicates(
        subset=["pretrained_sha256"]
    )
    unique_pretrained_count = len(pretrained)
    aggregate_rows = _aggregate_rows(
        unlearned,
        "unlearned_across_seeds",
        excluded={"seed"},
    )
    aggregate_rows.extend(
        _aggregate_rows(comparisons, "paired_comparison_across_seeds", excluded={"seed"})
    )
    if unique_pretrained_count == 1:
        aggregate_rows.extend(
            _aggregate_rows(
                pretrained,
                "shared_pretrained_reference_not_replicated",
                excluded={"seed"},
            )
        )
    else:
        aggregate_rows.extend(
            _aggregate_rows(
                pretrained,
                "unique_pretrained_checkpoints",
                excluded={"seed"},
            )
        )

    aggregate = pd.DataFrame(aggregate_rows)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "number_of_unlearning_runs": len(results),
        "number_of_unique_pretrained_checkpoints": unique_pretrained_count,
        "shared_pretrained_values_treated_as_independent": False,
        "rows": aggregate_rows,
    }
    files = {
        "per_seed": output_dir / "per_seed_summary.csv",
        "aggregate": output_dir / "multi_seed_summary.csv",
        "aggregate_json": output_dir / "multi_seed_summary.json",
    }
    per_seed.to_csv(files["per_seed"], index=False, float_format="%.12g")
    aggregate.to_csv(files["aggregate"], index=False, float_format="%.12g")
    _write_json(files["aggregate_json"], payload)
    return output_dir, files, aggregate


def _log_aggregate_to_wandb(
    aggregate: pd.DataFrame,
    files: Mapping[str, Path],
    results: Sequence[EvaluationResult],
    settings: RuntimeConfig,
    group: str,
) -> None:
    if settings.wandb_mode == "disabled":
        return
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("W&B aggregate logging requested but wandb is unavailable") from error

    first = results[0].context
    project = (
        settings.wandb_project
        or first.config.get("WANDB_PROJECT")
        or first.config.get("NAME", "fed-scrub")
    )
    run = None
    try:
        run = wandb.init(
            project=str(project),
            entity=settings.wandb_entity,
            name=_safe_name(f"yeom-aggregate-{group}"),
            group=group,
            job_type="yeom-aggregate",
            tags=["yeom", "membership-inference", "aggregate"] + list(
                settings.wandb_tags or []
            ),
            mode=settings.wandb_mode,
            config={
                "number_of_unlearning_runs": len(results),
                "number_of_unique_pretrained_checkpoints": len(
                    {result.context.pretrained_sha256 for result in results}
                ),
            },
            reinit=True,
        )
        scalars = {}
        for _, row in aggregate.iterrows():
            key = _safe_name(f"{row['section']}/{row['metric']}")
            scalars[f"yeom/aggregate/{key}/mean"] = row["mean"]
            if pd.notna(row["sample_std"]):
                scalars[f"yeom/aggregate/{key}/sample_std"] = row["sample_std"]
        run.log(scalars)
        run.log({"yeom/aggregate_table": wandb.Table(dataframe=aggregate)})

        artifact = wandb.Artifact(
            name=_safe_name(f"yeom-aggregate-{group}"),
            type="yeom-aggregate",
        )
        for path in files.values():
            artifact.add_file(str(path))
        run.log_artifact(artifact)
    finally:
        if run is not None:
            run.finish()


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", "none", "null"}:
        return False
    raise ValueError(f"Invalid Boolean value: {value!r}")


def _optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized or normalized.lower() in {"none", "null"}:
        return None
    return normalized


def _wandb_mode(value: Any) -> str:
    normalized = str(value or "online").strip().lower()
    aliases = {
        "on": "online",
        "true": "online",
        "1": "online",
        "off": "disabled",
        "false": "disabled",
        "0": "disabled",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"online", "offline", "disabled"}:
        raise ValueError(
            "YEOM_WANDB_MODE/WANDB_MODE must be online, offline, disabled, ON, or OFF"
        )
    return normalized


def _runtime_config(config: Mapping[str, Any]) -> RuntimeConfig:
    checkpoint_kind = str(config.get("YEOM_CHECKPOINT_KIND", "best")).strip().lower()
    if checkpoint_kind not in {"best", "latest"}:
        raise ValueError("YEOM_CHECKPOINT_KIND must be 'best' or 'latest'")

    forget_view = str(config.get("YEOM_FORGET_VIEW", "clean")).strip().lower()
    if forget_view not in {"clean", "training"}:
        raise ValueError("YEOM_FORGET_VIEW must be 'clean' or 'training'")

    scenario = _optional_string(config.get("YEOM_SCENARIO"))
    if scenario is not None and scenario not in {
        "client-level",
        "data-level",
        "unspecified",
    }:
        raise ValueError(
            "YEOM_SCENARIO must be client-level, data-level, or unspecified"
        )

    batch_size_raw = config.get("YEOM_BATCH_SIZE")
    batch_size = int(batch_size_raw) if batch_size_raw not in (None, "") else None
    num_workers = int(config.get("YEOM_NUM_WORKERS", 0))
    warning_drop = float(config.get("YEOM_UTILITY_WARNING_ACCURACY_DROP", 0.05))
    if batch_size is not None and batch_size <= 0:
        raise ValueError("YEOM_BATCH_SIZE must be positive")
    if num_workers < 0:
        raise ValueError("YEOM_NUM_WORKERS cannot be negative")
    if not 0 <= warning_drop <= 1:
        raise ValueError("YEOM_UTILITY_WARNING_ACCURACY_DROP must be in [0, 1]")

    tag_value = str(config.get("YEOM_WANDB_TAGS", "")).strip()
    tags = tuple(tag.strip() for tag in tag_value.split(",") if tag.strip())
    return RuntimeConfig(
        unlearned_checkpoint=_optional_string(
            config.get("YEOM_UNLEARNED_CHECKPOINT")
        ),
        checkpoint_kind=checkpoint_kind,
        index_dir=_optional_string(config.get("YEOM_INDEX_DIR")),
        output_dir=_optional_string(config.get("YEOM_OUTPUT_DIR")),
        aggregate_output_dir=_optional_string(
            config.get("YEOM_AGGREGATE_OUTPUT_DIR")
        ),
        data_root=_optional_string(config.get("YEOM_DATA_ROOT")),
        download=_parse_bool(config.get("YEOM_DOWNLOAD"), default=True),
        device=str(config.get("YEOM_DEVICE", "auto")).strip(),
        batch_size=batch_size,
        num_workers=num_workers,
        partition_seed=int(config.get("YEOM_PARTITION_SEED", 42)),
        forget_view=forget_view,
        backdoor_target=int(config.get("YEOM_BACKDOOR_TARGET", 0)),
        scenario=scenario,
        method=_optional_string(config.get("YEOM_METHOD")),
        utility_warning_accuracy_drop=warning_drop,
        wandb_mode=_wandb_mode(
            config.get("YEOM_WANDB_MODE", config.get("WANDB_MODE", "online"))
        ),
        wandb_project=_optional_string(
            config.get("YEOM_WANDB_PROJECT", config.get("WANDB_PROJECT"))
        ),
        wandb_entity=_optional_string(
            config.get("YEOM_WANDB_ENTITY", config.get("WANDB_ENTITY"))
        ),
        wandb_group=_optional_string(config.get("YEOM_WANDB_GROUP")),
        wandb_run_name=_optional_string(config.get("YEOM_WANDB_RUN_NAME")),
        wandb_tags=tags,
    )


def _discover_environment_directories(root: Path) -> list[Path]:
    if (root / ".env").is_file() and (root / ".env.training").is_file():
        return [root]

    directories = sorted(
        {
            env_file.parent
            for env_file in root.rglob(".env")
            if (env_file.parent / ".env.training").is_file()
        }
    )
    if not directories:
        raise FileNotFoundError(
            f"No directory containing both .env and .env.training was found under {root}"
        )
    return directories


def main() -> int:
    env_root_value = os.environ.get("EXP_ENV_DIR")
    if not env_root_value:
        raise EnvironmentError(
            "EXP_ENV_DIR is not set. Point it to an unlearning environment directory."
        )

    repository_root = _repository_root()
    env_root = _resolve_path(env_root_value, repository_root)
    _require_directory(env_root, "EXP_ENV_DIR")
    env_directories = _discover_environment_directories(env_root)
    print(f"Yeom evaluation environments: {[str(path) for path in env_directories]}")

    results: list[EvaluationResult] = []
    settings_by_environment: list[RuntimeConfig] = []
    default_group = _safe_name(f"yeom-{env_root.name}-{_utc_timestamp()}")
    multiple_runs = len(env_directories) > 1

    for env_dir in env_directories:
        config = load_config(str(env_dir))
        settings = _runtime_config(config)
        group = settings.wandb_group or default_group
        result = evaluate_environment(
            settings,
            env_dir,
            multiple_runs=multiple_runs,
        )
        print(f"Yeom outputs written to: {result.output_dir}")
        _log_result_to_wandb(result, settings, group)
        results.append(result)
        settings_by_environment.append(settings)

    if len(results) > 1:
        settings = settings_by_environment[0]
        group = settings.wandb_group or default_group
        aggregate_dir, aggregate_files, aggregate = _aggregate_results(
            results, settings
        )
        print(f"Multi-run Yeom summary written to: {aggregate_dir}")
        _log_aggregate_to_wandb(
            aggregate=aggregate,
            files=aggregate_files,
            results=results,
            settings=settings,
            group=group,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
