from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_NAME = "akariasai/PopQA"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "processed" / "popqa_clean.csv"
REQUIRED_COLUMNS = ["question", "possible_answers", "s_pop"]


def normalize_question(text: str) -> str:
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    return " ".join(text.split())


def relation_column(relation: str) -> str:
    return "rel_" + re.sub(r"[^a-z0-9]+", "_", relation.lower()).strip("_")


def inspect_schema(df: pd.DataFrame) -> None:
    print(f"Rows: {len(df)}")
    print("Columns and dtypes:")
    print(df.dtypes.to_string())
    print("Sample rows:")
    for _, row in df.head(3).iterrows():
        print(row.to_dict())


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["question"] = df["question"].replace(r"^\s*$", np.nan, regex=True)
    df["possible_answers"] = df["possible_answers"].map(
        lambda value: json.dumps(json.loads(value), ensure_ascii=False) if isinstance(value, str) else None
    )
    df.loc[df["possible_answers"] == "[]", "possible_answers"] = None
    before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS)
    print(f"Dropped {before - len(df)} rows with missing question, answer, or popularity.")

    before = len(df)
    df = df.assign(_norm=df["question"].map(normalize_question))
    df = df.sort_values(["s_pop", "id"], ascending=[False, True])
    df = df.drop_duplicates(subset="_norm", keep="first").drop(columns="_norm")
    df = df.sort_values("id").reset_index(drop=True)
    print(f"Removed {before - len(df)} duplicate-question rows.")

    df["s_pop_log"] = np.log10(df["s_pop"] + 1)
    one_hot = pd.get_dummies(df["prop"]).astype(int)
    one_hot.columns = [relation_column(name) for name in one_hot.columns]
    return pd.concat([df, one_hot[sorted(one_hot.columns)]], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output CSV path")
    args = parser.parse_args()

    from datasets import load_dataset

    raw = load_dataset(DATASET_NAME, split="test").to_pandas()
    inspect_schema(raw)
    cleaned = clean(raw)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.out, index=False)

    n_relations = sum(col.startswith("rel_") for col in cleaned.columns)
    pop = cleaned["s_pop"]
    print(f"Saved {args.out}")
    print(f"Rows: {len(raw)} -> {len(cleaned)}; relation types: {n_relations}")
    print(f"s_pop min/median/max: {pop.min()} / {pop.median()} / {pop.max()}")


if __name__ == "__main__":
    main()
