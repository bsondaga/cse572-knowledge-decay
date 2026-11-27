"""Run controlled rehearsal experiments using Task 6."""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "rehearsal"
SELECTIONS = RESULTS / "selections"
EXPERIMENTS = RESULTS / "experiments"
CHECKPOINTS = ROOT / "results" / "controlled_update" / "task11"

PRE_UPDATE = (
    ROOT / "data" / "processed" / "task11_pre_update_results.csv"
)

STRATEGIES = (
    "none", "random", "popanchor", "rarest", "risk_based"
)
BUDGETS = (5, 10, 20)
SEEDS = (0, 1, 2)


def run_name(strategy, budget, seed):
    if strategy == "none":
        return f"none_seed{seed}"
    return f"{strategy}_budget{budget}_seed{seed}"


def load_matches(path, column):
    df = pd.read_csv(path, dtype={"id": str})

    if len(df) != 285 or not df["id"].is_unique:
        raise ValueError(f"Invalid evaluation population: {path}")

    if not df[column].isin([0, 1]).all():
        raise ValueError(f"Invalid match values: {path}")

    return df.set_index("id")[column]


def run_experiment(strategy, budget, seed):
    name = run_name(strategy, budget, seed)
    output = EXPERIMENTS / f"{name}.csv"
    metrics = EXPERIMENTS / f"{name}.json"

    pre = load_matches(PRE_UPDATE, "baseline_exact_match")

    if not pre.eq(1).all():
        raise ValueError("Pre-update baseline is not fully correct")

    if output.exists() and metrics.exists():
        post = load_matches(output, "post_update_exact_match")

        if set(post.index) != set(pre.index):
            raise ValueError(f"ID mismatch: {output}")

        print(f"Skipping completed experiment: {name}", flush=True)
        return output

    command = [
        sys.executable,
        "-u",
        str(ROOT / "src" / "controlled_update.py"),
        "--model-name", "EleutherAI/pythia-160m",
        "--sample-size", "128",
        "--batch-size", "2",
        "--max-length", "128",
        "--epochs", "1",
        "--learning-rate", "1e-5",
        "--seed", str(seed),
        "--output-dir", str(CHECKPOINTS / name),
        "--only-known",
        "--known-facts-count", "285",
        "--pre-update-results", str(PRE_UPDATE),
        "--post-update-results", str(output),
        "--post-update-metrics", str(metrics),
    ]

    if strategy != "none":
        selection = SELECTIONS / f"{name}.csv"

        if not selection.is_file():
            raise FileNotFoundError(selection)

        command.extend([
            "--rehearsal-file", str(selection),
            "--facts-file",
            str(ROOT / "data" / "processed" / "popqa_clean.csv"),
        ])

    print(f"\nStarting experiment: {name}", flush=True)

    subprocess.run(command, cwd=ROOT, check=True)

    post = load_matches(output, "post_update_exact_match")

    if set(post.index) != set(pre.index):
        raise ValueError(f"ID mismatch: {output}")

    return output


def calculate_metrics(strategy, budget, seed):
    name = run_name(strategy, budget, seed)

    baseline = load_matches(
        EXPERIMENTS / f"none_seed{seed}.csv",
        "post_update_exact_match",
    )

    post = load_matches(
        EXPERIMENTS / f"{name}.csv",
        "post_update_exact_match",
    )

    if set(baseline.index) != set(post.index):
        raise ValueError("Baseline and experiment IDs differ")

    post = post.reindex(baseline.index)

    selected = set()

    if strategy != "none":
        selection = pd.read_csv(
            SELECTIONS / f"{name}.csv",
            dtype={"fact_id": str},
        )

        if not selection["fact_id"].is_unique:
            raise ValueError("Duplicate rehearsal IDs")

        selected = set(selection["fact_id"])

        if len(selected) != 285 * budget // 100:
            raise ValueError("Incorrect rehearsal budget")

    if not selected.issubset(set(post.index)):
        raise ValueError("Unknown rehearsal IDs")

    mask = ~post.index.isin(selected)

    baseline_rate = float(1 - baseline.mean())
    forgetting_rate = float(1 - post.mean())

    baseline_non = float(1 - baseline.loc[mask].mean())
    forgetting_non = float(1 - post.loc[mask].mean())

    return {
        "strategy": strategy,
        "budget_pct": budget,
        "seed": seed,
        "known_facts": len(post),
        "rehearsed_facts": len(selected),
        "non_rehearsed_facts": int(mask.sum()),
        "forgotten_all": int((post == 0).sum()),
        "forgetting_rate_all": forgetting_rate,
        "baseline_forgetting_rate_all": baseline_rate,
        "reduction_all_pp": (
            baseline_rate - forgetting_rate
        ) * 100,
        "forgotten_non_rehearsed": int(
            (post.loc[mask] == 0).sum()
        ),
        "forgetting_rate_non_rehearsed": forgetting_non,
        "baseline_forgetting_rate_non_rehearsed": baseline_non,
        "reduction_non_rehearsed_pp": (
            baseline_non - forgetting_non
        ) * 100,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--strategies", nargs="+",
        choices=STRATEGIES, default=list(STRATEGIES),
    )

    parser.add_argument(
        "--budgets", nargs="+", type=int,
        choices=BUDGETS, default=list(BUDGETS),
    )

    parser.add_argument(
        "--seeds", nargs="+", type=int,
        choices=SEEDS, default=list(SEEDS),
    )

    parser.add_argument(
        "--output", type=Path,
        default=RESULTS / "experiment_results.csv",
    )

    args = parser.parse_args()

    EXPERIMENTS.mkdir(parents=True, exist_ok=True)

    rows = []

    for seed in args.seeds:
        run_experiment("none", 5, seed)

        for budget in args.budgets:
            for strategy in args.strategies:
                run_experiment(strategy, budget, seed)

                row = calculate_metrics(strategy, budget, seed)
                rows.append(row)

                args.output.parent.mkdir(
                    parents=True, exist_ok=True
                )

                pd.DataFrame(rows).to_csv(
                    args.output, index=False
                )

                print(
                    f"{strategy}, budget={budget}, seed={seed}: "
                    f"forgetting={row['forgetting_rate_all']:.2%}, "
                    f"reduction={row['reduction_all_pp']:.2f} pp",
                    flush=True,
                )

    print(f"\nSaved {len(rows)} configurations to {args.output}")


if __name__ == "__main__":
    main()