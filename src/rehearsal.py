"""Selection-only rehearsal strategies for PopQA facts."""

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