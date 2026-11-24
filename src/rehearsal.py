"""Selection-only rehearsal strategies for PopQA facts."""

from pathlib import Path

import pandas as pd


def _validate(df: pd.DataFrame, k: int) -> None:
    """Validate common inputs for rehearsal selection."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("facts must be a pandas DataFrame")
    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError("k must be an integer")
    if k < 0 or k > len(df):
        raise ValueError("k must be between 0 and the number of facts")
    if "id" not in df.columns:
        raise ValueError("Missing required column: id")
    if df["id"].isna().any() or df["id"].duplicated().any():
        raise ValueError("Fact ids must be unique and non-null")


def select_none(df: pd.DataFrame) -> pd.DataFrame:
    """No rehearsal: select no facts."""
    return df.iloc[0:0].copy()


def select_random(
    df: pd.DataFrame, k: int, seed: int = 42
) -> pd.DataFrame:
    """Select k facts randomly and reproducibly."""
    _validate(df, k)
    return df.sample(n=k, random_state=seed).reset_index(drop=True)


def select_popanchor(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Select k facts with the highest subject popularity."""
    _validate(df, k)
    if "s_pop" not in df.columns:
        raise ValueError("Missing required column: s_pop")
    if df["s_pop"].isna().any():
        raise ValueError("Popularity scores must not be missing")

    return (
        df.sort_values(
            by=["s_pop", "id"],
            ascending=[False, True],
            kind="stable",
        )
        .head(k)
        .reset_index(drop=True)
    )


def select_risk_based(
    df: pd.DataFrame,
    k: int,
    risk_column: str = "forgetting_risk",
) -> pd.DataFrame:
    """Select k facts with the highest predicted forgetting risk."""
    _validate(df, k)
    if risk_column not in df.columns:
        raise ValueError(f"Missing required column: {risk_column}")
    if df[risk_column].isna().any():
        raise ValueError("Risk scores must not be missing")

    return (
        df.sort_values(
            by=[risk_column, "id"],
            ascending=[False, True],
            kind="stable",
        )
        .head(k)
        .reset_index(drop=True)
    )


def select_rarest(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Select k facts with the lowest subject popularity."""
    _validate(df, k)
    if "s_pop" not in df.columns:
        raise ValueError("Missing required column: s_pop")
    if df["s_pop"].isna().any():
        raise ValueError("Popularity scores must not be missing")

    return (
        df.sort_values(
            by=["s_pop", "id"],
            ascending=[True, True],
            kind="stable",
        )
        .head(k)
        .reset_index(drop=True)
    )


BUDGETS = (5, 10, 20)

SELECTION_DIR = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "rehearsal"
    / "selections"
)


def budget_to_k(n_known: int, budget_pct: int) -> int:
    """Convert a percentage budget to a number of known facts."""
    if not isinstance(n_known, int) or isinstance(n_known, bool):
        raise TypeError("n_known must be an integer")
    if n_known < 0:
        raise ValueError("n_known must not be negative")
    if budget_pct not in BUDGETS:
        raise ValueError("Budget must be 5, 10 or 20 percent")

    return n_known * budget_pct // 100


def normalize_facts(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize analysis-table columns for rehearsal selection."""
    facts = df.copy()

    if "id" not in facts.columns and "fact_id" in facts.columns:
        facts = facts.rename(columns={"fact_id": "id"})

    if "s_pop" not in facts.columns and "popularity" in facts.columns:
        facts = facts.rename(columns={"popularity": "s_pop"})

    _validate(facts, 0)
    return facts


def generate_selection(
    df: pd.DataFrame,
    strategy: str,
    budget_pct: int,
    seed: int = 42,
    output_dir: Path = SELECTION_DIR,
) -> Path:
    """Select known facts and save their IDs to a CSV file."""
    facts = normalize_facts(df)
    k = budget_to_k(len(facts), budget_pct)

    if strategy == "none":
        selected = select_none(facts)
    elif strategy == "random":
        selected = select_random(facts, k, seed)
    elif strategy == "popanchor":
        selected = select_popanchor(facts, k)
    elif strategy == "rarest":
        selected = select_rarest(facts, k)
    elif strategy == "risk_based":
        selected = select_risk_based(facts, k)
    else:
        raise ValueError(f"Unknown rehearsal strategy: {strategy}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_dir
        / f"{strategy}_budget{budget_pct}_seed{seed}.csv"
    )

    selected[["id"]].rename(
        columns={"id": "fact_id"}
    ).to_csv(output_path, index=False)

    return output_path
