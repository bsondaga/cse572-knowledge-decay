import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.real_label_classifiers import (
    CV_METRIC_COLUMNS,
    CV_SCORE_COLUMNS,
    CV_SUMMARY_COLUMNS,
    PREDICTOR_COLUMNS,
    FrozenSplitIndices,
    PER_SEED_COLUMNS,
    SUMMARY_COLUMNS,
    CrossValidationFold,
    aggregate_seed_results,
    aggregate_cv_metrics,
    build_final_risk_scores,
    evaluate_test_scores,
    fit_predict_indices,
    fit_predict_test_scores,
    group_rare_relations,
    load_frozen_dataset,
    load_frozen_split,
    make_cv_folds,
    make_lightgbm_classifier,
    make_logistic_pipeline,
    map_split_indices,
    prevalence_test_scores,
    run_oof_experiment,
    select_final_risk_model,
    run_experiments,
    validate_oof_score_artifact,
    select_population,
    select_predictor_features,
    select_popularity_features,
    validate_frozen_dataset,
)
from src.rehearsal import load_risk_scores

RELATION_COLUMNS = tuple(column for column in PREDICTOR_COLUMNS if column.startswith("rel_"))


def make_dataset() -> pd.DataFrame:
    ids = [9001, 23, 501, 44, 66]
    guessable = [0, 1, 0, 0, 1]
    rows = []
    for index, fact_id in enumerate(ids):
        row = {
            "id": fact_id,
            "relation": "author",
            "subject_pop_log": 1.0 + index,
            "object_pop_log": 2.0 + index,
            "subject_pop": 100 + index,
            "object_pop": 50 + index,
            "degree": index,
            "guess_count": index,
            "guessable": guessable[index],
            "pop_bucket": index % 4,
            "s_pop_z": float(index),
            "forgotten_lr1e5": index % 2,
            "forgotten_default": (index + 1) % 2,
        }
        row.update(
            {
                column: int(column == RELATION_COLUMNS[index % len(RELATION_COLUMNS)])
                for column in RELATION_COLUMNS
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def make_assignments(dataset: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [44, 501, 9001, 66, 23],
            "split": ["test", "val", "train", "val", "train"],
        }
    )


def make_experiment_dataset(row_count=60) -> pd.DataFrame:
    rows = []
    for index in range(row_count):
        row = {
            "id": 10000 + index * 17,
            "relation": "author" if index % 2 else "country",
            "subject_pop_log": float(index),
            "object_pop_log": float(index % 7),
            "subject_pop": 10 + index,
            "object_pop": 20 + index,
            "degree": index % 5,
            "guess_count": index % 3,
            "guessable": int(index % 4 == 3),
            "pop_bucket": index % 4,
            "s_pop_z": float(index),
            "forgotten_lr1e5": index % 2,
            "forgotten_default": (index + 1) % 2,
        }
        row.update(
            {
                column: int(column == RELATION_COLUMNS[index % len(RELATION_COLUMNS)])
                for column in RELATION_COLUMNS
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def make_cv_dataset() -> pd.DataFrame:
    relations = (
        ["author"] * 40
        + ["country"] * 40
        + ["color"] * 4
        + ["composer"] * 4
        + ["director"] * 4
        + ["mother"] * 4
        + ["father"] * 4
    )
    relation_feature = {
        relation: f"rel_{relation.replace(' ', '_')}" for relation in set(relations)
    }
    rows = []
    for index, relation in enumerate(relations):
        row = {
            "id": 80000 + index * 13,
            "relation": relation,
            "subject_pop_log": float(index % 17),
            "object_pop_log": float(index % 11),
            "subject_pop": 100 + index,
            "object_pop": 200 + index,
            "degree": index % 8,
            "guess_count": index % 4,
            "guessable": 0,
            "pop_bucket": index % 4,
            "s_pop_z": float(index),
            "forgotten_lr1e5": index % 2,
            "forgotten_default": (index + 1) % 2,
        }
        row.update(
            {
                column: int(column == relation_feature.get(relation))
                for column in RELATION_COLUMNS
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def make_commit2_summary(lr_auprc=0.7963, lgbm_auprc=0.7806):
    return pd.DataFrame(
        [
            {"population": "known", "model": "logistic_regression_full", "auprc_mean": lr_auprc},
            {"population": "known", "model": "lightgbm_full", "auprc_mean": lgbm_auprc},
            {"population": "all", "model": "lightgbm_full", "auprc_mean": 0.9},
        ]
    )


def make_oof_artifact(ids):
    rows = []
    for model in ("logistic_regression_full", "lightgbm_full"):
        for seed in (0, 1, 2):
            for index, fact_id in enumerate(ids):
                rows.append(
                    {
                        "id": fact_id,
                        "population": "known",
                        "model": model,
                        "seed": seed,
                        "fold": index % 5,
                        "true_label": index % 2,
                        "forgetting_risk": (index + seed + (model == "lightgbm_full")) / 20,
                    }
                )
    return pd.DataFrame(rows, columns=CV_SCORE_COLUMNS)


class SpyProbabilityModel:
    classes_ = np.array([0, 1])

    def __init__(self):
        self.train_features = None
        self.train_labels = None
        self.test_features = None

    def fit(self, features, labels):
        self.train_features = features.copy()
        self.train_labels = np.array(labels, copy=True)
        return self

    def predict_proba(self, features):
        self.test_features = features.copy()
        return np.tile([0.3, 0.7], (len(features), 1))


class FrozenDatasetTests(unittest.TestCase):
    def test_loads_valid_dataset_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "real_dataset.csv"
            make_dataset().to_csv(path, index=False)

            dataset = load_frozen_dataset(path)

        self.assertEqual(len(dataset), 5)
        self.assertEqual(set(dataset["forgotten_lr1e5"].unique()), {0, 1})

    def test_selects_known_and_all_populations(self):
        dataset = validate_frozen_dataset(make_dataset())

        known = select_population(dataset, "known")
        all_rows = select_population(dataset, "all")

        self.assertEqual(known["id"].tolist(), [9001, 501, 44])
        self.assertTrue((known["guessable"] == 0).all())
        self.assertEqual(all_rows["id"].tolist(), dataset["id"].tolist())

    def test_selects_only_established_predictors(self):
        dataset = validate_frozen_dataset(make_dataset())

        features = select_predictor_features(dataset)

        self.assertEqual(tuple(features.columns), PREDICTOR_COLUMNS)
        self.assertEqual(features.shape[1], 18)
        self.assertNotIn("id", features.columns)
        self.assertNotIn("relation", features.columns)
        self.assertNotIn("guessable", features.columns)
        self.assertNotIn("forgotten_lr1e5", features.columns)
        self.assertNotIn("forgotten_default", features.columns)
        self.assertNotIn("guess_count", features.columns)
        self.assertNotIn("degree", features.columns)

    def test_duplicate_dataset_ids_fail(self):
        dataset = make_dataset()
        dataset.loc[1, "id"] = dataset.loc[0, "id"]

        with self.assertRaisesRegex(ValueError, "duplicate fact IDs"):
            validate_frozen_dataset(dataset)

    def test_missing_dataset_ids_fail(self):
        dataset = make_dataset()
        dataset.loc[0, "id"] = np.nan

        with self.assertRaisesRegex(ValueError, "missing fact IDs"):
            validate_frozen_dataset(dataset)

    def test_non_binary_target_fails(self):
        dataset = make_dataset()
        dataset.loc[0, "forgotten_lr1e5"] = 2

        with self.assertRaisesRegex(ValueError, "forgotten_lr1e5.*binary"):
            validate_frozen_dataset(dataset)

    def test_invalid_guessable_fails(self):
        dataset = make_dataset()
        dataset.loc[0, "guessable"] = -1

        with self.assertRaisesRegex(ValueError, "guessable.*binary"):
            validate_frozen_dataset(dataset)

    def test_non_finite_predictor_fails(self):
        dataset = make_dataset()
        dataset.loc[0, "subject_pop_log"] = np.inf

        with self.assertRaisesRegex(ValueError, "subject_pop_log.*finite"):
            validate_frozen_dataset(dataset)

    def test_non_numeric_predictor_fails(self):
        dataset = make_dataset()
        dataset["object_pop_log"] = dataset["object_pop_log"].astype(object)
        dataset.loc[0, "object_pop_log"] = "unknown"

        with self.assertRaisesRegex(ValueError, "object_pop_log.*numeric"):
            validate_frozen_dataset(dataset)

    def test_missing_required_column_fails(self):
        dataset = make_dataset().drop(columns="rel_sport")

        with self.assertRaisesRegex(ValueError, "missing required columns"):
            validate_frozen_dataset(dataset)

    def test_empty_relation_fails(self):
        dataset = make_dataset()
        dataset.loc[0, "relation"] = "  "

        with self.assertRaisesRegex(ValueError, "relation.*non-empty"):
            validate_frozen_dataset(dataset)


class FrozenSplitTests(unittest.TestCase):
    def setUp(self):
        self.dataset = validate_frozen_dataset(make_dataset())
        self.assignments = make_assignments(self.dataset)

    def test_loads_valid_split_and_maps_by_id_not_file_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            split_dir = Path(temp_dir) / "all"
            split_dir.mkdir(parents=True)
            self.assignments.to_csv(split_dir / "seed0.csv", index=False)

            indices = load_frozen_split(
                self.dataset, "all", seed=0, splits_dir=Path(temp_dir)
            )

        self.assertEqual(indices.train.tolist(), [0, 1])
        self.assertEqual(indices.validation.tolist(), [2, 4])
        self.assertEqual(indices.test.tolist(), [3])

    def test_partitions_are_disjoint_and_cover_population(self):
        indices = map_split_indices(self.dataset, self.assignments)
        combined = np.concatenate([indices.train, indices.validation, indices.test])

        self.assertEqual(len(np.unique(combined)), len(self.dataset))
        np.testing.assert_array_equal(np.sort(combined), np.arange(len(self.dataset)))
        indices.validate(len(self.dataset))

    def test_duplicate_split_ids_fail(self):
        assignments = pd.concat(
            [self.assignments, self.assignments.iloc[[0]]], ignore_index=True
        )

        with self.assertRaisesRegex(ValueError, "duplicate fact IDs"):
            map_split_indices(self.dataset, assignments)

    def test_missing_split_ids_fail(self):
        with self.assertRaisesRegex(ValueError, "missing .* selected population IDs"):
            map_split_indices(self.dataset, self.assignments.iloc[:-1])

    def test_extra_split_ids_fail(self):
        assignments = pd.concat(
            [
                self.assignments,
                pd.DataFrame({"id": [123456], "split": ["train"]}),
            ],
            ignore_index=True,
        )

        with self.assertRaisesRegex(ValueError, "outside the selected population"):
            map_split_indices(self.dataset, assignments)

    def test_invalid_membership_fails(self):
        assignments = self.assignments.copy()
        assignments.loc[0, "split"] = "validation"

        with self.assertRaisesRegex(ValueError, "invalid membership"):
            map_split_indices(self.dataset, assignments)

    def test_empty_partition_fails(self):
        assignments = self.assignments.copy()
        assignments["split"] = "train"

        with self.assertRaisesRegex(ValueError, "each be non-empty"):
            map_split_indices(self.dataset, assignments)

    def test_rejects_unsupported_population_or_seed(self):
        with self.assertRaisesRegex(ValueError, "population must be one of"):
            load_frozen_split(self.dataset, "unknown", seed=0)
        with self.assertRaisesRegex(ValueError, "seed must be one of"):
            load_frozen_split(self.dataset, "all", seed=3)


class FrozenModelEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.dataset = validate_frozen_dataset(make_experiment_dataset(20))
        self.features = select_predictor_features(self.dataset)
        self.labels = self.dataset["forgotten_lr1e5"].to_numpy()
        self.split = FrozenSplitIndices(
            train=np.arange(12),
            validation=np.arange(12, 16),
            test=np.arange(16, 20),
        )

    def test_fit_helper_uses_only_frozen_train_and_test_rows(self):
        model = SpyProbabilityModel()

        scores = fit_predict_test_scores(
            model, self.features, self.labels, self.split
        )

        pd.testing.assert_frame_equal(
            model.train_features, self.features.iloc[self.split.train]
        )
        np.testing.assert_array_equal(model.train_labels, self.labels[self.split.train])
        pd.testing.assert_frame_equal(
            model.test_features, self.features.iloc[self.split.test]
        )
        self.assertEqual(len(scores), len(self.split.test))
        self.assertTrue(np.all((scores >= 0) & (scores <= 1)))

    def test_logistic_regression_pipeline_scales_train_rows_only(self):
        features = self.features.copy()
        features.loc[self.split.test, "subject_pop_log"] = 10000.0
        model = make_logistic_pipeline(seed=13)

        scores = fit_predict_test_scores(model, features, self.labels, self.split)

        scaler = model.named_steps["scaler"]
        np.testing.assert_allclose(
            scaler.mean_, features.iloc[self.split.train].mean(axis=0).to_numpy()
        )
        self.assertEqual(len(scores), len(self.split.test))
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(np.all((scores >= 0) & (scores <= 1)))

    def test_lightgbm_uses_fixed_settings_and_returns_test_probabilities(self):
        model = make_lightgbm_classifier(seed=13)
        params = model.get_params()
        self.assertEqual(
            {key: params[key] for key in (
                "n_estimators", "learning_rate", "max_depth", "num_leaves",
                "class_weight", "random_state", "n_jobs", "verbosity",
            )},
            {
                "n_estimators": 100,
                "learning_rate": 0.05,
                "max_depth": 4,
                "num_leaves": 15,
                "class_weight": "balanced",
                "random_state": 13,
                "n_jobs": 1,
                "verbosity": -1,
            },
        )

        scores = fit_predict_test_scores(
            model, self.features, self.labels, self.split
        )

        self.assertEqual(len(scores), len(self.split.test))
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(np.all((scores >= 0) & (scores <= 1)))

    def test_popularity_selector_returns_exactly_two_features(self):
        features = select_popularity_features(self.dataset)

        self.assertEqual(list(features.columns), ["subject_pop_log", "object_pop_log"])

    def test_prevalence_baseline_uses_train_labels_only(self):
        train_labels = self.labels[self.split.train]
        first_scores = prevalence_test_scores(train_labels, len(self.split.test))
        changed_test_labels = self.labels.copy()
        changed_test_labels[self.split.test] = 1 - changed_test_labels[self.split.test]
        second_scores = prevalence_test_scores(
            changed_test_labels[self.split.train], len(self.split.test)
        )

        np.testing.assert_array_equal(first_scores, second_scores)
        np.testing.assert_array_equal(
            first_scores, np.full(len(self.split.test), train_labels.mean())
        )

    def test_metrics_receive_test_labels_only_and_have_requested_keys(self):
        test_scores = np.array([0.9, 0.7, 0.2, 0.1])
        with patch("src.real_label_classifiers.evaluate_predictions") as evaluate:
            evaluate.return_value = {
                "auprc": 1.0,
                "f1": 1.0,
                "recall_at_k": {1: 0.5, 5: 1.0, 10: 1.0},
            }
            metrics = evaluate_test_scores(self.labels, self.split, test_scores)

        np.testing.assert_array_equal(
            evaluate.call_args.args[0], self.labels[self.split.test]
        )
        np.testing.assert_array_equal(evaluate.call_args.args[1], test_scores)
        self.assertEqual(
            set(metrics), {"auprc", "f1", "recall_at_1", "recall_at_5", "recall_at_10"}
        )
        self.assertEqual(evaluate.call_args.kwargs["threshold"], 0.5)
        self.assertEqual(evaluate.call_args.kwargs["k_values"], (1, 5, 10))

    def test_aggregate_reports_mean_and_sample_standard_deviation(self):
        rows = []
        for seed, auprc in enumerate((0.2, 0.4, 0.6)):
            rows.append(
                {
                    "population": "known",
                    "seed": seed,
                    "model": "logistic_regression_full",
                    "n_train": 10,
                    "n_val": 2,
                    "n_test": 3,
                    "train_prevalence": 0.4,
                    "auprc": auprc,
                    "f1": auprc,
                    "recall_at_1": auprc,
                    "recall_at_5": auprc,
                    "recall_at_10": auprc,
                }
            )

        summary = aggregate_seed_results(pd.DataFrame(rows))

        self.assertEqual(tuple(summary.columns), SUMMARY_COLUMNS)
        self.assertAlmostEqual(summary.loc[0, "auprc_mean"], 0.4)
        self.assertAlmostEqual(summary.loc[0, "auprc_std"], 0.2)

    def test_run_experiments_honors_frozen_population_and_seed_assignments(self):
        dataset = validate_frozen_dataset(make_experiment_dataset(60))
        with tempfile.TemporaryDirectory() as temp_dir:
            split_root = Path(temp_dir)
            expected_counts = {}
            for population in ("known", "all"):
                selected = select_population(dataset, population)
                population_dir = split_root / population
                population_dir.mkdir()
                for seed in (0, 1):
                    train_count = int(len(selected) * (0.6 if seed == 0 else 0.5))
                    validation_count = int(len(selected) * 0.2)
                    memberships = (
                        ["train"] * train_count
                        + ["val"] * validation_count
                        + ["test"] * (len(selected) - train_count - validation_count)
                    )
                    assignments = pd.DataFrame(
                        {"id": selected["id"], "split": memberships}
                    ).iloc[::-1]
                    assignments.to_csv(population_dir / f"seed{seed}.csv", index=False)
                    expected_counts[(population, seed)] = (
                        train_count,
                        validation_count,
                        len(selected) - train_count - validation_count,
                    )

            captured = []

            def fake_fit_predict(model, features, labels, split):
                captured.append((len(labels), split.train.copy(), split.test.copy()))
                return np.full(len(split.test), 0.6)

            with patch(
                "src.real_label_classifiers.fit_predict_test_scores",
                side_effect=fake_fit_predict,
            ):
                per_seed, summary = run_experiments(
                    dataset,
                    splits_dir=split_root,
                    populations=("all", "known"),
                    seeds=(0, 1),
                )

        self.assertEqual(tuple(per_seed.columns), PER_SEED_COLUMNS)
        self.assertEqual(per_seed["population"].drop_duplicates().tolist(), ["known", "all"])
        self.assertEqual(len(per_seed), 2 * 2 * 4)
        self.assertEqual(len(summary), 2 * 4)
        for _, row in per_seed.drop_duplicates(["population", "seed"]).iterrows():
            self.assertEqual(
                (row["n_train"], row["n_val"], row["n_test"]),
                expected_counts[(row["population"], row["seed"])],
            )
        self.assertEqual(len(captured), 2 * 2 * 3)


class CrossValidationTests(unittest.TestCase):
    def setUp(self):
        self.dataset = validate_frozen_dataset(make_cv_dataset())
        self.labels = self.dataset["forgotten_lr1e5"].to_numpy()
        self.relations = self.dataset["relation"].to_numpy()

    def test_task1_rare_relation_grouping_makes_joint_strata_valid(self):
        groups = group_rare_relations(self.labels, self.relations)
        self.assertEqual(set(groups[-20:]), {"other"})
        counts = pd.DataFrame({"label": self.labels, "relation_group": groups}).groupby(
            ["label", "relation_group"]
        ).size()
        self.assertTrue((counts >= 5).all())

    def test_undersized_joint_stratum_fails_instead_of_falling_back(self):
        labels = np.array([0] * 6 + [1] * 4)
        relations = np.array(["one"] * 10)

        with self.assertRaisesRegex(ValueError, "joint .*strata.*at least 5"):
            make_cv_folds(labels, relations, seed=0)

    def test_five_fold_assignment_is_deterministic_for_same_seed(self):
        first = make_cv_folds(self.labels, self.relations, seed=1)
        second = make_cv_folds(self.labels, self.relations, seed=1)

        self.assertEqual(len(first), 5)
        for fold_a, fold_b in zip(first, second):
            np.testing.assert_array_equal(fold_a.train, fold_b.train)
            np.testing.assert_array_equal(fold_a.validation, fold_b.validation)

    def test_every_fact_is_oof_once_and_folds_do_not_overlap(self):
        folds = make_cv_folds(self.labels, self.relations, seed=2)
        validation_counts = np.zeros(len(self.labels), dtype=int)

        for fold in folds:
            fold.validate(len(self.labels))
            self.assertEqual(np.intersect1d(fold.train, fold.validation).size, 0)
            validation_counts[fold.validation] += 1

        np.testing.assert_array_equal(validation_counts, np.ones(len(self.labels), dtype=int))

    def test_fit_and_scaling_use_only_training_fold_rows(self):
        features = select_predictor_features(self.dataset).copy()
        fold = make_cv_folds(self.labels, self.relations, seed=0)[0]
        features.iloc[fold.validation, 0] = 100000.0
        model = make_logistic_pipeline(seed=0)

        scores = fit_predict_indices(
            model, features, self.labels, fold.train, fold.validation
        )

        np.testing.assert_allclose(
            model.named_steps["scaler"].mean_,
            features.iloc[fold.train].mean(axis=0).to_numpy(),
        )
        self.assertEqual(len(scores), len(fold.validation))
        self.assertTrue(np.all((scores >= 0) & (scores <= 1)))

    def test_oof_experiment_artifact_coverage_schemas_and_model_choice(self):
        def fake_fit_predict(model, features, labels, train_indices, validation_indices):
            probability = 0.8 if isinstance(model, Pipeline) else 0.2
            return np.full(len(validation_indices), probability)

        with patch(
            "src.real_label_classifiers.fit_predict_indices",
            side_effect=fake_fit_predict,
        ):
            oof_scores, fold_metrics, summary, final_scores, final_model = run_oof_experiment(
                self.dataset,
                commit2_summary=make_commit2_summary(),
            )

        self.assertEqual(final_model, "logistic_regression_full")
        self.assertEqual(tuple(oof_scores.columns), CV_SCORE_COLUMNS)
        self.assertEqual(tuple(fold_metrics.columns), CV_METRIC_COLUMNS)
        self.assertEqual(tuple(summary.columns), CV_SUMMARY_COLUMNS)
        self.assertEqual(len(oof_scores), len(self.dataset) * 3 * 2)
        self.assertEqual(len(fold_metrics), 3 * 5 * 2)
        self.assertEqual(len(summary), 2)
        validate_oof_score_artifact(oof_scores, self.dataset["id"], (0, 1, 2))
        self.assertEqual(len(final_scores), len(self.dataset))
        self.assertEqual(list(final_scores.columns), ["id", "forgetting_risk"])
        self.assertEqual(final_scores["id"].nunique(), len(self.dataset))
        self.assertTrue(final_scores["forgetting_risk"].between(0, 1).all())
        np.testing.assert_allclose(final_scores["forgetting_risk"], 0.8)

    def test_commit2_known_test_auprc_selects_logistic_not_cv_scores(self):
        self.assertEqual(
            select_final_risk_model(make_commit2_summary()),
            "logistic_regression_full",
        )
        self.assertEqual(
            select_final_risk_model(make_commit2_summary(0.7, 0.8)),
            "lightgbm_full",
        )

    def test_final_risk_averages_exactly_three_seed_scores_per_id(self):
        ids = [12, 88, 104, 305]
        scores = make_oof_artifact(ids)

        final = build_final_risk_scores(
            scores, ids, "logistic_regression_full", seeds=(0, 1, 2)
        )

        self.assertEqual(list(final.columns), ["id", "forgetting_risk"])
        self.assertEqual(final["id"].tolist(), sorted(ids))
        expected = {
            fact_id: np.mean(
                [
                    (index + seed) / 20
                    for seed in (0, 1, 2)
                ]
            )
            for index, fact_id in enumerate(ids)
        }
        self.assertEqual(final["id"].nunique(), len(ids))
        self.assertEqual(set(final["id"]), set(ids))
        for row in final.itertuples(index=False):
            self.assertAlmostEqual(row.forgetting_risk, expected[row.id])

    def test_rehearsal_loader_accepts_exact_final_risk_schema_and_coverage(self):
        ids = [12, 88, 104, 305]
        final = build_final_risk_scores(
            make_oof_artifact(ids), ids, "logistic_regression_full"
        )
        facts = pd.DataFrame({"id": ids, "s_pop": [10, 20, 30, 40]})
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "risk_scores.csv"
            final.to_csv(path, index=False)

            merged = load_risk_scores(facts, path)

        self.assertEqual(len(merged), len(ids))
        self.assertEqual(merged["id"].nunique(), len(ids))
        self.assertEqual(set(merged["id"]), set(ids))
        self.assertTrue(merged["forgetting_risk"].between(0, 1).all())

    def test_cv_summary_mean_and_sample_std_are_hand_calculated(self):
        fold_metrics = pd.DataFrame(
            [
                {
                    "population": "known",
                    "model": "logistic_regression_full",
                    "seed": seed,
                    "fold": fold,
                    "n_train": 8,
                    "n_validation": 2,
                    "auprc": value,
                    "f1": value,
                    "recall_at_1": value,
                    "recall_at_5": value,
                    "recall_at_10": value,
                }
                for seed, value in enumerate((0.2, 0.4, 0.6))
                for fold in [0]
            ],
            columns=CV_METRIC_COLUMNS,
        )

        summary = aggregate_cv_metrics(fold_metrics)

        self.assertEqual(tuple(summary.columns), CV_SUMMARY_COLUMNS)
        self.assertAlmostEqual(summary.loc[0, "auprc_mean"], 0.4)
        self.assertAlmostEqual(summary.loc[0, "auprc_std"], 0.2)


if __name__ == "__main__":
    unittest.main()