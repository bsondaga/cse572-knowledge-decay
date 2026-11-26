from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from src.make_real_dataset import FROZEN_DATASET_PATH, PROCESSED
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.make_real_dataset import FROZEN_DATASET_PATH, PROCESSED

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
ENRICHED_PATH = ROOT / "results" / "frozen" / "real_dataset_enriched.csv"
STATS_PATH = ROOT / "results" / "feature_enrichment_stats.md"

# Relation -> entity type of the answer's subject/topic; unknown relations -> "other".
ENTITY_TYPES = {
    "occupation": "person",
    "place of birth": "person",
    "father": "person",
    "mother": "person",
    "religion": "person",
    "country": "place",
    "capital": "place",
    "capital of": "place",
    "director": "creative_work",
    "screenwriter": "creative_work",
    "producer": "creative_work",
    "genre": "creative_work",
    "composer": "creative_work",
    "author": "creative_work",
    "sport": "other",
    "color": "other",
}
TYPE_NAMES = ("person", "place", "creative_work", "other")
NEW_BASE = ["question_length", "question_chars", "answer_length", "answer_chars", "entity_type"]
PRE_SKIP = {"id", "question", "prediction", "possible_answers", "baseline_exact_match"}


def _first_answer(raw: object) -> str:
    try:
        answers = json.loads(raw)
    except (TypeError, ValueError):
        return ""
    return str(answers[0]) if isinstance(answers, list) and answers else ""


def enrich(base: pd.DataFrame, clean: pd.DataFrame, pre: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add text-length and entity-type features; base columns and row order stay unchanged."""
    text = clean[["id", "question", "possible_answers"]].drop_duplicates("id")
    merged = base[["id"]].merge(text, on="id", how="left", validate="one_to_one")
    if merged["question"].isna().any():
        raise ValueError("some facts have no question in the cleaned PopQA file")

    new = pd.DataFrame({"id": base["id"].to_numpy()})
    question = merged["question"].astype(str)
    answer = merged["possible_answers"].map(_first_answer)
    new["question_length"] = question.str.split().str.len().to_numpy()
    new["question_chars"] = question.str.len().to_numpy()
    new["answer_length"] = answer.str.split().str.len().to_numpy()
    new["answer_chars"] = answer.str.len().to_numpy()
    new["entity_type"] = base["relation"].map(ENTITY_TYPES).fillna("other").to_numpy()
    for name in TYPE_NAMES:
        new[f"et_{name}"] = (new["entity_type"] == name).astype(int)

    if pre is not None and "confidence" in pre.columns:
        keep = ["id"] + [c for c in pre.columns if c not in PRE_SKIP and c.startswith("confidence")]
        new = new.merge(pre[keep].drop_duplicates("id"), on="id", how="left", validate="one_to_one")

    out = pd.concat([base.reset_index(drop=True), new.drop(columns="id")], axis=1)
    if len(out) != len(base) or not out["id"].equals(base["id"].reset_index(drop=True)):
        raise ValueError("enriched table does not keep the input ids and order")
    return out


def new_columns(enriched: pd.DataFrame, base: pd.DataFrame) -> list[str]:
    return [c for c in enriched.columns if c not in base.columns and c != "entity_type"]


def stats_markdown(enriched: pd.DataFrame, base: pd.DataFrame) -> str:
    cols = new_columns(enriched, base)
    lines = ["# Feature enrichment stats", "",
             "Spearman correlation of each new feature with `forgotten_lr1e5`.", ""]
    for name, pop in (("known (guessable == 0)", enriched[enriched["guessable"] == 0]),
                      ("all", enriched)):
        lines += [f"## {name}: {len(pop)} rows", "",
                  "| feature | missing | spearman rho | p-value |", "|---|---|---|---|"]
        for col in cols + ["entity_type"]:
            missing = int(pop[col].isna().sum())
            if col == "entity_type":
                rho = p = float("nan")
            else:
                rho, p = spearmanr(pop[col], pop["forgotten_lr1e5"], nan_policy="omit")
            fmt = (lambda v: "n/a" if pd.isna(v) else f"{v:.3f}")
            lines.append(f"| {col} | {missing} | {fmt(rho)} | {fmt(p)} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the enriched feature table.")
    parser.add_argument("--base", type=Path, default=FROZEN_DATASET_PATH)
    parser.add_argument("--clean", type=Path, default=PROCESSED / "popqa_clean.csv")
    parser.add_argument("--pre", type=Path, default=PROCESSED / "pre_update_results.csv")
    parser.add_argument("--out", type=Path, default=ENRICHED_PATH)
    parser.add_argument("--stats", type=Path, default=STATS_PATH)
    args = parser.parse_args()

    base = pd.read_csv(args.base)
    pre = pd.read_csv(args.pre) if args.pre.exists() else None
    enriched = enrich(base, pd.read_csv(args.clean), pre)
    labels = pd.read_csv(args.base.parent / "labels.csv")
    if enriched["id"].nunique() != len(enriched) or set(enriched["id"]) != set(labels["id"]):
        raise ValueError("enriched ids differ from labels.csv")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.out, index=False)
    args.stats.write_text(stats_markdown(enriched, base) + "\n", encoding="utf-8")
    has_conf = "confidence" in enriched.columns
    print(f"rows: {len(enriched)}, new columns: {new_columns(enriched, base)}")
    print(f"confidence merged: {has_conf}")


if __name__ == "__main__":
    main()
