from __future__ import annotations

import copy
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

try:
    from src.evaluation_metrics import evaluate_predictions
    from src.graphsage_model import DEFAULT_EDGES, DEFAULT_FEATURES, GraphSAGEModel, build_graph_data
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.evaluation_metrics import evaluate_predictions
    from src.graphsage_model import DEFAULT_EDGES, DEFAULT_FEATURES, GraphSAGEModel, build_graph_data

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS_PATH = ROOT / "results" / "frozen" / "real_dataset.csv"
DEFAULT_SPLITS_DIR = ROOT / "results" / "splits"
KNOWN_POPULATION = "known"
TARGET_COLUMN = "forgotten_lr1e5"
SUPPORTED_SEEDS = (0, 1, 2)
SPLIT_NAMES = ("train", "val", "test")
EMBEDDING_DIM = 16
RESULTS_DIR = ROOT / "results"
METRICS_BY_SEED_OUTPUT = RESULTS_DIR / "task8_graphsage_metrics_by_seed.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "task8_graphsage_summary.csv"
SCORES_OUTPUT = RESULTS_DIR / "task8_graphsage_scores.csv"
LOSS_CURVES_OUTPUT = RESULTS_DIR / "task8_graphsage_loss_curves.csv"
METRIC_NAMES = ("auprc", "f1", "recall_at_1", "recall_at_5", "recall_at_10")
METRICS_BY_SEED_COLUMNS = (
    "population",
    "seed",
    "model",
    "n_train",
    "n_validation",
    "n_test",
    "best_epoch",
    "best_validation_loss",
    "stopped_early",
    "pos_weight",
    *METRIC_NAMES,
)
SUMMARY_COLUMNS = (
    "population",
    "model",
    "n_seeds",
    *(f"{metric}_{stat}" for metric in METRIC_NAMES for stat in ("mean", "std")),
)
SCORES_COLUMNS = (
    "population",
    "seed",
    "id",
    "split",
    "true_label",
    "forgetting_risk",
)
LOSS_CURVE_COLUMNS = (
    "seed",
    "epoch",
    "train_loss",
    "validation_loss",
    "is_best",
    "stopped_early",
)


@dataclass(frozen=True)
class GraphSAGETrainingConfig:
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    max_epochs: int = 300
    patience: int = 20
    min_delta: float = 1e-4

    def validate(self) -> None:
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if (
            isinstance(self.max_epochs, bool)
            or not isinstance(self.max_epochs, (int, np.integer))
            or self.max_epochs < 1
        ):
            raise ValueError("max_epochs must be a positive integer")
        if (
            isinstance(self.patience, bool)
            or not isinstance(self.patience, (int, np.integer))
            or self.patience < 1
        ):
            raise ValueError("patience must be a positive integer")
        if not np.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("min_delta must be finite and non-negative")


@dataclass
class GraphSAGETrainingResult:
    model: nn.Module
    best_epoch: int
    best_validation_loss: float
    stopped_early: bool
    loss_history: list[dict[str, int | float | bool]]
    pos_weight: float


class GraphSAGEClassifier(nn.Module):
    """GraphSAGE embedding encoder followed by a single forgotten-class logit."""

    def __init__(
        self,
        encoder: nn.Module | None = None,
        embedding_dim: int = EMBEDDING_DIM,
    ) -> None:
        super().__init__()
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        self.encoder = encoder or GraphSAGEModel(output_dim=embedding_dim)
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        embeddings = self.encoder(x, edge_index)
        logits = self.classifier(embeddings).squeeze(-1)
        if logits.ndim != 1 or logits.shape[0] != x.shape[0]:
            raise ValueError("GraphSAGE classifier must return one logit per graph node")
        if not torch.isfinite(logits).all():
            raise ValueError("GraphSAGE classifier produced non-finite logits")
        return logits


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable CPU/GPU runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def compute_train_pos_weight(masks: GraphLabelMasks) -> Tensor:
    """Compute retained/forgotten weight using only frozen training labels."""
    masks.validate(len(masks.labels))
    train_mask = masks.train_mask
    if not torch.any(train_mask & masks.labeled_mask):
        raise ValueError("train mask must contain labeled nodes")
    if torch.any(train_mask & ~masks.labeled_mask):
        raise ValueError("train mask must select only labeled nodes")

    train_labels = masks.labels[train_mask]
    if not torch.isfinite(train_labels).all() or not torch.all(
        (train_labels == 0) | (train_labels == 1)
    ):
        raise ValueError("training labels must contain only finite binary 0/1 values")
    positives = int((train_labels == 1).sum())
    negatives = int((train_labels == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("training labels must contain both forgotten and retained classes")

    weight = negatives / positives
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError("training pos_weight must be finite and positive")
    return torch.tensor(weight, dtype=torch.float32, device=masks.labels.device)


def masked_bce_loss(
    logits: Tensor,
    labels: Tensor,
    mask: Tensor,
    criterion: nn.BCEWithLogitsLoss,
) -> Tensor:
    """Compute BCE only for the selected labeled-node mask."""
    if logits.ndim != 1 or labels.ndim != 1 or mask.ndim != 1:
        raise ValueError("logits, labels, and mask must be one-dimensional")
    if len(logits) != len(labels) or len(mask) != len(labels):
        raise ValueError("logits, labels, and mask must have equal lengths")
    if mask.dtype != torch.bool:
        raise ValueError("supervised loss mask must use torch.bool")
    if not torch.any(mask):
        raise ValueError("supervised loss mask must select at least one node")
    selected_labels = labels[mask]
    if not torch.isfinite(selected_labels).all() or not torch.all(
        (selected_labels == 0) | (selected_labels == 1)
    ):
        raise ValueError("supervised labels must contain only finite binary 0/1 values")
    return criterion(logits[mask], selected_labels)


def train_graphsage(
    graph,
    masks: GraphLabelMasks,
    seed: int,
    config: GraphSAGETrainingConfig | None = None,
    model: nn.Module | None = None,
) -> GraphSAGETrainingResult:
    """Train on the full graph with train-mask loss and validation-only stopping."""
    config = config or GraphSAGETrainingConfig()
    config.validate()
    if not hasattr(graph, "x") or not hasattr(graph, "edge_index"):
        raise ValueError("graph must provide x and edge_index tensors")
    if graph.x.ndim != 2 or graph.edge_index.ndim != 2 or graph.edge_index.shape[0] != 2:
        raise ValueError("graph requires x [num_nodes, features] and edge_index [2, edges]")
    num_nodes = graph.x.shape[0]
    masks.validate(num_nodes)
    seed_everything(seed)

    device = graph.x.device
    labels = masks.labels.to(device)
    train_mask = masks.train_mask.to(device)
    validation_mask = masks.validation_mask.to(device)
    # Validate that training targets contain both classes before allocating the model.
    active_masks = GraphLabelMasks(
        labels=labels,
        labeled_mask=masks.labeled_mask.to(device),
        train_mask=train_mask,
        validation_mask=validation_mask,
        test_mask=masks.test_mask.to(device),
        node_ids=masks.node_ids,
    )
    pos_weight = compute_train_pos_weight(active_masks)

    classifier = (model or GraphSAGEClassifier()).to(device)
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, int | float | bool]] = []
    stopped_early = False

    for epoch in range(1, config.max_epochs + 1):
        classifier.train()
        optimizer.zero_grad(set_to_none=True)
        train_logits = classifier(graph.x, graph.edge_index)
        train_loss = masked_bce_loss(train_logits, labels, train_mask, criterion)
        train_loss.backward()
        optimizer.step()

        classifier.eval()
        with torch.no_grad():
            validation_logits = classifier(graph.x, graph.edge_index)
            validation_loss = masked_bce_loss(
                validation_logits, labels, validation_mask, criterion
            )

        train_loss_value = float(train_loss.detach().cpu())
        validation_loss_value = float(validation_loss.detach().cpu())
        if not np.isfinite(train_loss_value) or not np.isfinite(validation_loss_value):
            raise ValueError("training or validation loss became non-finite")

        is_best = validation_loss_value < best_validation_loss - config.min_delta
        if is_best:
            best_validation_loss = validation_loss_value
            best_epoch = epoch
            best_state = {
                name: value.detach().clone()
                for name, value in classifier.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_value,
                "validation_loss": validation_loss_value,
                "is_best": is_best,
            }
        )
        if epochs_without_improvement >= config.patience:
            stopped_early = epoch < config.max_epochs
            break

    if best_state is None:
        raise RuntimeError("training did not produce a finite best validation checkpoint")
    classifier.load_state_dict(best_state)
    return GraphSAGETrainingResult(
        model=classifier,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        stopped_early=stopped_early,
        loss_history=history,
        pos_weight=float(pos_weight.detach().cpu()),
    )


def predict_graph_probabilities(
    model: nn.Module,
    graph,
    mask: Tensor | None = None,
) -> Tensor:
    """Run full-graph inference, sigmoid logits, and optionally select node scores."""
    if not hasattr(graph, "x") or not hasattr(graph, "edge_index"):
        raise ValueError("graph must provide x and edge_index tensors")
    model.eval()
    with torch.no_grad():
        logits = model(graph.x, graph.edge_index)
        if logits.ndim != 1 or logits.shape[0] != graph.x.shape[0]:
            raise ValueError("model must return one logit per graph node")
        probabilities = torch.sigmoid(logits)

    if not torch.isfinite(probabilities).all() or torch.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError("predicted probabilities must be finite and in [0, 1]")
    if mask is None:
        return probabilities
    if mask.ndim != 1 or len(mask) != graph.x.shape[0] or mask.dtype != torch.bool:
        raise ValueError("prediction mask must be torch.bool with length num_graph_nodes")
    return probabilities[mask.to(probabilities.device)]


@dataclass(frozen=True)
class GraphLabelMasks:
    """Full-graph labels and frozen split masks, indexed in graph node order."""

    labels: Tensor
    labeled_mask: Tensor
    train_mask: Tensor
    validation_mask: Tensor
    test_mask: Tensor
    node_ids: tuple[object, ...]

    def validate(self, num_nodes: int) -> None:
        tensors = (
            self.labels,
            self.labeled_mask,
            self.train_mask,
            self.validation_mask,
            self.test_mask,
        )
        if any(tensor.ndim != 1 or len(tensor) != num_nodes for tensor in tensors):
            raise ValueError("labels and masks must be one-dimensional with length num_nodes")
        if self.labels.dtype != torch.float32:
            raise ValueError("labels must use torch.float32")
        if any(
            mask.dtype != torch.bool
            for mask in (
                self.labeled_mask,
                self.train_mask,
                self.validation_mask,
                self.test_mask,
            )
        ):
            raise ValueError("labeled and split masks must use torch.bool")
        if len(self.node_ids) != num_nodes or len(set(self.node_ids)) != num_nodes:
            raise ValueError("node_ids must contain one unique fact ID per graph node")

        supervised = (self.train_mask, self.validation_mask, self.test_mask)
        if any(torch.any(mask & ~self.labeled_mask) for mask in supervised):
            raise ValueError("split masks must select only labeled graph nodes")
        if any(
            torch.any(left & right)
            for index, left in enumerate(supervised)
            for right in supervised[index + 1 :]
        ):
            raise ValueError("train, validation, and test masks must be disjoint")
        if not torch.equal(
            self.train_mask | self.validation_mask | self.test_mask,
            self.labeled_mask,
        ):
            raise ValueError("train, validation, and test masks must cover labeled nodes")
        if any(not torch.any(mask) for mask in supervised):
            raise ValueError("train, validation, and test masks must all be non-empty")


def load_known_labels(path: Path = DEFAULT_LABELS_PATH) -> pd.DataFrame:
    """Load, validate, and return only known fact IDs with the Task 8 target."""
    try:
        dataset = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        raise ValueError(f"Unable to load frozen labels from {path}: {error}") from error
    if dataset.empty:
        raise ValueError("frozen label dataset must not be empty")

    required = {"id", "guessable", TARGET_COLUMN}
    missing = required - set(dataset.columns)
    if missing:
        raise ValueError(f"Frozen label dataset is missing required columns: {sorted(missing)}")
    if dataset["id"].isna().any():
        raise ValueError("Frozen label dataset contains missing fact IDs")
    duplicate_ids = dataset.loc[dataset["id"].duplicated(), "id"].unique().tolist()
    if duplicate_ids:
        raise ValueError(f"Frozen label dataset contains duplicate fact IDs: {duplicate_ids[:5]}")

    known = dataset.copy()
    for column in ("guessable", TARGET_COLUMN):
        try:
            values = pd.to_numeric(known[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"{column} must contain only binary 0/1 values") from error
        numeric = values.to_numpy(dtype=float)
        if not np.isfinite(numeric).all() or not np.isin(numeric, [0, 1]).all():
            raise ValueError(f"{column} must contain only binary 0/1 values")
        known[column] = values.astype(int)

    known = known.loc[known["guessable"] == 0, ["id", TARGET_COLUMN]].copy()
    if known.empty:
        raise ValueError("frozen dataset contains no known facts (guessable == 0)")
    return known.reset_index(drop=True)


def validate_split_assignments(
    assignments: pd.DataFrame, known_ids
) -> pd.DataFrame:
    """Validate frozen ID-to-split rows against the exact known ID set."""
    if not isinstance(assignments, pd.DataFrame):
        raise TypeError("split assignments must be a pandas DataFrame")
    if assignments.empty:
        raise ValueError("frozen split assignments must not be empty")
    missing_columns = {"id", "split"} - set(assignments.columns)
    if missing_columns:
        raise ValueError(f"Frozen split is missing required columns: {sorted(missing_columns)}")
    if assignments["id"].isna().any():
        raise ValueError("Frozen split contains missing fact IDs")
    duplicate_ids = assignments.loc[
        assignments["id"].duplicated(), "id"
    ].unique().tolist()
    if duplicate_ids:
        raise ValueError(f"Frozen split contains duplicate IDs: {duplicate_ids[:5]}")
    if assignments["split"].isna().any():
        raise ValueError("Frozen split contains missing membership values")

    invalid_memberships = sorted(set(assignments["split"].astype(str)) - set(SPLIT_NAMES))
    if invalid_memberships:
        raise ValueError(f"Frozen split contains invalid membership values: {invalid_memberships}")

    known_id_set = set(known_ids)
    split_id_set = set(assignments["id"].tolist())
    missing_ids = known_id_set - split_id_set
    extra_ids = split_id_set - known_id_set
    if missing_ids:
        raise ValueError(
            f"Frozen split is missing {len(missing_ids)} known fact IDs: "
            f"{sorted(missing_ids, key=str)[:5]}"
        )
    if extra_ids:
        raise ValueError(
            f"Frozen split contains {len(extra_ids)} IDs outside the known population: "
            f"{sorted(extra_ids, key=str)[:5]}"
        )
    empty_partitions = [
        name for name in SPLIT_NAMES if not (assignments["split"] == name).any()
    ]
    if empty_partitions:
        raise ValueError(f"Frozen split partitions must be non-empty: {empty_partitions}")
    return assignments.loc[:, ["id", "split"]].copy().reset_index(drop=True)


def load_frozen_split_assignments(
    seed: int,
    known_ids,
    splits_dir: Path = DEFAULT_SPLITS_DIR,
) -> pd.DataFrame:
    """Load one authoritative known-population split; never generate assignments."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed not in SUPPORTED_SEEDS:
        raise ValueError(f"seed must be one of {SUPPORTED_SEEDS}")
    path = Path(splits_dir) / KNOWN_POPULATION / f"seed{seed}.csv"
    try:
        assignments = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        raise ValueError(f"Unable to load frozen split from {path}: {error}") from error
    return validate_split_assignments(assignments, known_ids)


def build_graph_label_masks(
    id_to_index: dict[object, int],
    known_labels: pd.DataFrame,
    assignments: pd.DataFrame,
    num_nodes: int,
) -> GraphLabelMasks:
    """Map known targets and frozen split memberships onto full-graph positions."""
    if isinstance(num_nodes, bool) or not isinstance(num_nodes, (int, np.integer)) or num_nodes < 1:
        raise ValueError("num_nodes must be a positive integer")
    if len(id_to_index) != num_nodes:
        raise ValueError("id_to_index must contain exactly one entry per graph node")
    if set(id_to_index.values()) != set(range(num_nodes)):
        raise ValueError("id_to_index values must be contiguous indices from 0 to num_nodes - 1")
    if known_labels.empty:
        raise ValueError("known labels must not be empty")
    missing_columns = {"id", TARGET_COLUMN} - set(known_labels.columns)
    if missing_columns:
        raise ValueError(f"Known labels are missing required columns: {sorted(missing_columns)}")
    if known_labels["id"].isna().any():
        raise ValueError("Known labels contain missing fact IDs")
    if known_labels["id"].duplicated().any():
        raise ValueError("Known labels contain duplicate IDs")

    try:
        target_values = pd.to_numeric(known_labels[TARGET_COLUMN], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{TARGET_COLUMN} must contain only binary 0/1 values") from error
    target_array = target_values.to_numpy(dtype=float)
    if not np.isfinite(target_array).all() or not np.isin(target_array, [0, 1]).all():
        raise ValueError(f"{TARGET_COLUMN} must contain only binary 0/1 values")

    known_ids = known_labels["id"].tolist()
    unknown_ids = [fact_id for fact_id in known_ids if fact_id not in id_to_index]
    if unknown_ids:
        raise ValueError(f"Known fact IDs are not present in the graph: {unknown_ids[:5]}")
    assignments = validate_split_assignments(assignments, known_ids)
    split_by_id = dict(zip(assignments["id"], assignments["split"]))

    labels = torch.zeros(num_nodes, dtype=torch.float32)
    labeled_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    validation_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask_by_split = {
        "train": train_mask,
        "val": validation_mask,
        "test": test_mask,
    }
    for fact_id, target in zip(known_ids, target_array):
        node_index = id_to_index[fact_id]
        labels[node_index] = float(target)
        labeled_mask[node_index] = True
        mask_by_split[split_by_id[fact_id]][node_index] = True

    inverse_mapping: list[object | None] = [None] * num_nodes
    for fact_id, node_index in id_to_index.items():
        inverse_mapping[node_index] = fact_id
    if any(fact_id is None for fact_id in inverse_mapping):
        raise ValueError("id_to_index does not map every graph node")

    masks = GraphLabelMasks(
        labels=labels,
        labeled_mask=labeled_mask,
        train_mask=train_mask,
        validation_mask=validation_mask,
        test_mask=test_mask,
        node_ids=tuple(inverse_mapping),
    )
    masks.validate(num_nodes)
    return masks


def load_graph_label_masks(
    id_to_index: dict[object, int],
    num_nodes: int,
    seed: int,
    labels_path: Path = DEFAULT_LABELS_PATH,
    splits_dir: Path = DEFAULT_SPLITS_DIR,
) -> GraphLabelMasks:
    """Load the known labels and frozen seed split, then map them to graph masks."""
    known_labels = load_known_labels(labels_path)
    assignments = load_frozen_split_assignments(
        seed, known_labels["id"].tolist(), splits_dir=splits_dir
    )
    return build_graph_label_masks(
        id_to_index, known_labels, assignments, num_nodes=num_nodes
    )


def summarize_graphsage_metrics(metrics_by_seed: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the three frozen-seed metric rows using sample standard deviation."""
    missing_columns = set(METRICS_BY_SEED_COLUMNS) - set(metrics_by_seed.columns)
    if missing_columns:
        raise ValueError(f"Task 8 metrics are missing columns: {sorted(missing_columns)}")
    if metrics_by_seed.empty:
        raise ValueError("Task 8 metrics must not be empty")

    rows = []
    for (population, model_name), group in metrics_by_seed.groupby(
        ["population", "model"], sort=False
    ):
        row: dict[str, int | float | str] = {
            "population": population,
            "model": model_name,
            "n_seeds": group["seed"].nunique(),
        }
        for metric in METRIC_NAMES:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        rows.append(row)

    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    return summary.sort_values(["population", "model"], kind="stable").reset_index(
        drop=True
    )


def run_graphsage_experiments(
    graph=None,
    id_to_index: dict[object, int] | None = None,
    masks_by_seed: dict[int, GraphLabelMasks] | None = None,
    *,
    features_path: Path = DEFAULT_FEATURES,
    edges_path: Path = DEFAULT_EDGES,
    labels_path: Path = DEFAULT_LABELS_PATH,
    splits_dir: Path = DEFAULT_SPLITS_DIR,
    seeds: tuple[int, ...] = SUPPORTED_SEEDS,
    config: GraphSAGETrainingConfig | None = None,
    training_function=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train the real known population for each frozen seed and return four tables."""
    if tuple(seeds) != SUPPORTED_SEEDS:
        raise ValueError(f"Task 8 experiment requires exactly seeds {SUPPORTED_SEEDS}")
    if graph is None:
        if id_to_index is not None:
            raise ValueError("id_to_index must only be supplied together with graph")
        graph_features = pd.read_csv(features_path)
        graph_edges = pd.read_csv(edges_path)
        graph, id_to_index = build_graph_data(graph_features, graph_edges)
    elif id_to_index is None:
        raise ValueError("id_to_index is required when graph is supplied")
    if graph.x.shape[0] != len(id_to_index):
        raise ValueError("graph node count does not match id_to_index")
    if masks_by_seed is not None and set(masks_by_seed) != set(seeds):
        raise ValueError(f"masks_by_seed must contain exactly seeds {seeds}")

    train_fn = training_function or train_graphsage
    metric_records: list[dict[str, int | float | bool | str]] = []
    score_records: list[dict[str, int | float | str]] = []
    loss_records: list[dict[str, int | float | bool]] = []

    for seed in seeds:
        masks = (
            masks_by_seed[seed]
            if masks_by_seed is not None
            else load_graph_label_masks(
                id_to_index,
                graph.x.shape[0],
                seed,
                labels_path=labels_path,
                splits_dir=splits_dir,
            )
        )
        masks.validate(graph.x.shape[0])
        result = train_fn(graph, masks, seed=seed, config=config)
        test_probabilities = predict_graph_probabilities(
            result.model, graph, mask=masks.test_mask
        )
        test_labels = masks.labels[masks.test_mask].detach().cpu().numpy()
        test_scores = test_probabilities.detach().cpu().numpy()
        metric_values = evaluate_predictions(
            test_labels,
            test_scores,
            threshold=0.5,
            k_values=(1, 5, 10),
        )
        metric_records.append(
            {
                "population": KNOWN_POPULATION,
                "seed": seed,
                "model": "graphsage",
                "n_train": int(masks.train_mask.sum()),
                "n_validation": int(masks.validation_mask.sum()),
                "n_test": int(masks.test_mask.sum()),
                "best_epoch": result.best_epoch,
                "best_validation_loss": result.best_validation_loss,
                "stopped_early": result.stopped_early,
                "pos_weight": result.pos_weight,
                "auprc": metric_values["auprc"],
                "f1": metric_values["f1"],
                "recall_at_1": metric_values["recall_at_k"][1],
                "recall_at_5": metric_values["recall_at_k"][5],
                "recall_at_10": metric_values["recall_at_k"][10],
            }
        )

        test_node_indices = torch.where(masks.test_mask.detach().cpu())[0].tolist()
        if len(test_node_indices) != len(test_scores):
            raise ValueError("test probability count does not match frozen test IDs")
        score_records.extend(
            {
                "population": KNOWN_POPULATION,
                "seed": seed,
                "id": masks.node_ids[node_index],
                "split": "test",
                "true_label": int(masks.labels[node_index]),
                "forgetting_risk": float(probability),
            }
            for node_index, probability in zip(test_node_indices, test_scores)
        )

        if not result.loss_history:
            raise ValueError(f"seed {seed}: training returned empty loss history")
        for position, history_row in enumerate(result.loss_history):
            train_loss = float(history_row["train_loss"])
            validation_loss = float(history_row["validation_loss"])
            if not np.isfinite(train_loss) or not np.isfinite(validation_loss):
                raise ValueError(f"seed {seed}: loss history contains non-finite values")
            loss_records.append(
                {
                    "seed": seed,
                    "epoch": int(history_row["epoch"]),
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "is_best": bool(history_row["is_best"]),
                    "stopped_early": bool(
                        result.stopped_early and position == len(result.loss_history) - 1
                    ),
                }
            )

    metrics_by_seed = pd.DataFrame(metric_records, columns=METRICS_BY_SEED_COLUMNS)
    metrics_by_seed = metrics_by_seed.sort_values("seed", kind="stable").reset_index(
        drop=True
    )
    summary = summarize_graphsage_metrics(metrics_by_seed)
    scores = pd.DataFrame(score_records, columns=SCORES_COLUMNS)
    scores = scores.sort_values(["seed", "id"], kind="stable").reset_index(drop=True)
    loss_curves = pd.DataFrame(loss_records, columns=LOSS_CURVE_COLUMNS)
    loss_curves = loss_curves.sort_values(["seed", "epoch"], kind="stable").reset_index(
        drop=True
    )
    return metrics_by_seed, summary, scores, loss_curves


def write_graphsage_outputs(
    metrics_by_seed: pd.DataFrame,
    summary: pd.DataFrame,
    scores: pd.DataFrame,
    loss_curves: pd.DataFrame,
    output_dir: Path = RESULTS_DIR,
) -> dict[str, Path]:
    """Write exactly the four approved Task 8 result CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics_by_seed": output_dir / METRICS_BY_SEED_OUTPUT.name,
        "summary": output_dir / SUMMARY_OUTPUT.name,
        "scores": output_dir / SCORES_OUTPUT.name,
        "loss_curves": output_dir / LOSS_CURVES_OUTPUT.name,
    }
    metrics_by_seed.to_csv(paths["metrics_by_seed"], index=False)
    summary.to_csv(paths["summary"], index=False)
    scores.to_csv(paths["scores"], index=False)
    loss_curves.to_csv(paths["loss_curves"], index=False)
    return paths


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train GraphSAGE on the frozen known facts.")
    parser.add_argument(
        "--run-experiment",
        action="store_true",
        help="train all three frozen seeds and write the four Task 8 result tables",
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--edges", type=Path, default=DEFAULT_EDGES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    args = parser.parse_args()
    if not args.run_experiment:
        parser.print_help()
        return

    metrics, summary, scores, loss_curves = run_graphsage_experiments(
        features_path=args.features,
        edges_path=args.edges,
        labels_path=args.labels,
        splits_dir=args.splits_dir,
    )
    paths = write_graphsage_outputs(metrics, summary, scores, loss_curves)
    for row in metrics.itertuples(index=False):
        print(
            f"seed={row.seed} best_epoch={row.best_epoch} "
            f"AUPRC={row.auprc:.4f} F1={row.f1:.4f} "
            f"recall@1={row.recall_at_1:.4f} recall@5={row.recall_at_5:.4f} "
            f"recall@10={row.recall_at_10:.4f}"
        )
    for output_path in paths.values():
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()