import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import src.final_predictors as fp
from src.enrich_features import ENTITY_TYPES
from src.final_predictors import (
    ENRICHED_PREDICTOR_COLUMNS,
    METRIC_RECORD_COLUMNS,
    MODEL_FACTORIES,
    ORIGINAL_PREDICTOR_COLUMNS,
    TASK7_ADDED_COLUMNS,
    compare_with_task5_baseline,
    load_enriched_dataset,
    metric_record,
    metric_records_frame,
    predictor_columns,
    run_frozen_seed,
    select_enriched_predictors,
    select_feature_set,
    select_known_population,
    task5_baseline_from_frame,
    validate_enriched_dataset,
    validate_predictor_columns,
)
from src.real_label_classifiers import (
    METRIC_NAMES,
    PREDICTOR_COLUMNS,
    RELATION_FEATURES,
    TARGET_COLUMN,
    make_lightgbm_classifier,
    make_logistic_pipeline,
)

RELATIONS = ["capital", "country", "religion", "place of birth", "genre", "sport", "father", "author"]
N_KNOWN = 40
N_GUESSABLE = 8


def make_enriched_dataset(seed=0):
    """Synthetic enriched table: 40 known + 8 guessable rows, IDs not in sorted order."""
    rng = np.random.default_rng(seed)
    n = N_KNOWN + N_GUESSABLE
    ids = rng.permutation(np.arange(1000, 1000 + n * 7, 7))
    rows = []
    for index, fact_id in enumerate(ids):
        relation = RELATIONS[index % len(RELATIONS)]
        entity = ENTITY_TYPES.get(relation, "other")
        label = int((index % 3 == 0) or (relation in {"religion", "genre"}))
        row = {
            "id": int(fact_id),
            "relation": relation,
            "subject_pop": int(rng.integers(1, 10_000)),
            "object_pop": int(rng.integers(1, 10_000)),
            "subject_pop_log": float(rng.normal(3, 1)),
            "object_pop_log": float(rng.normal(4, 1)),
            "degree": int(rng.integers(0, 5)),
            "guess_count": 0 if index < N_KNOWN else 2,
            "guessable": 0 if index < N_KNOWN else 1,
            "pop_bucket": int(index % 4),
            "s_pop_z": float(rng.normal()),
            TARGET_COLUMN: label,
            "forgotten_default": 1 - label,
            "question_length": int(rng.integers(4, 12)),
            "question_chars": int(rng.integers(20, 80)),
            "answer_length": int(rng.integers(1, 4)),
            "answer_chars": int(rng.integers(3, 30)),
            "entity_type": entity,
        }
        for column in RELATION_FEATURES:
            row[column] = int(column == "rel_" + relation.replace(" ", "_"))
        for name in ("person", "place", "creative_work", "other"):
            row[f"et_{name}"] = int(entity == name)
        rows.append(row)
    return pd.DataFrame(rows)


def write_splits(dataset, directory):
    """Write frozen known splits for seeds 0-2, rows shuffled relative to the dataset."""
    known_ids = dataset.loc[dataset["guessable"] == 0, "id"].to_numpy()
    for seed in (0, 1, 2):
        rng = np.random.default_rng(100 + seed)
        order = rng.permutation(known_ids)
        names = ["train"] * 28 + ["val"] * 6 + ["test"] * 6
        frame = pd.DataFrame({"id": order, "split": names})
        frame = frame.sample(frac=1.0, random_state=seed)  # file row order != dataset order
        path = Path(directory) / "known"
        path.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path / f"seed{seed}.csv", index=False)


class SplitDirMixin:
    def setUp(self):
        self.dataset = make_enriched_dataset()
        self._tmp = tempfile.TemporaryDirectory()
        self.splits_dir = Path(self._tmp.name)
        write_splits(self.dataset, self.splits_dir)

    def tearDown(self):
        self._tmp.cleanup()

    def assignments(self, seed):
        return pd.read_csv(self.splits_dir / "known" / f"seed{seed}.csv")


class PredictorSetTests(unittest.TestCase):
    def test_original_set_is_exact_task5_set(self):
        self.assertEqual(len(ORIGINAL_PREDICTOR_COLUMNS), 18)
        self.assertEqual(ORIGINAL_PREDICTOR_COLUMNS, tuple(PREDICTOR_COLUMNS))
        self.assertEqual(predictor_columns("original"), tuple(PREDICTOR_COLUMNS))

    def test_enriched_set_order_and_additions(self):
        expected_added = (
            "question_length", "question_chars", "answer_length", "answer_chars",
            "et_person", "et_place", "et_creative_work", "et_other",
        )
        self.assertEqual(TASK7_ADDED_COLUMNS, expected_added)
        self.assertEqual(len(ENRICHED_PREDICTOR_COLUMNS), 26)
        self.assertEqual(ENRICHED_PREDICTOR_COLUMNS, (*PREDICTOR_COLUMNS, *expected_added))
        self.assertEqual(predictor_columns("enriched"), ENRICHED_PREDICTOR_COLUMNS)
        self.assertEqual(len(set(ENRICHED_PREDICTOR_COLUMNS)), 26)

    def test_confidence_and_non_predictors_absent(self):
        excluded = {
            "confidence", "id", "relation", "entity_type", TARGET_COLUMN, "forgotten_default",
            "guessable", "guess_count", "degree", "subject_pop", "object_pop", "pop_bucket",
            "s_pop_z", "forgetting_risk",
        }
        self.assertFalse(excluded & set(ENRICHED_PREDICTOR_COLUMNS))
        self.assertFalse(any(c.startswith("pop_bucket") or c.endswith("_z")
                             for c in ENRICHED_PREDICTOR_COLUMNS))

    def test_selector_returns_exact_columns_in_order(self):
        dataset = make_enriched_dataset()
        shuffled = dataset[dataset.columns[::-1]]
        selected = select_enriched_predictors(shuffled)
        self.assertEqual(list(selected.columns), list(ENRICHED_PREDICTOR_COLUMNS))
        self.assertEqual(list(select_feature_set(shuffled, "original").columns),
                         list(PREDICTOR_COLUMNS))

    def test_unknown_feature_set_raises(self):
        with self.assertRaises(ValueError):
            predictor_columns("everything")


class LeakageGuardTests(unittest.TestCase):
    def assert_rejected(self, column):
        with self.assertRaisesRegex(ValueError, "prohibited predictor"):
            validate_predictor_columns([*PREDICTOR_COLUMNS, column])

    def test_prohibited_columns_raise(self):
        for column in ("id", "relation", "entity_type", TARGET_COLUMN, "forgotten_default",
                       "guessable", "guess_count", "degree", "subject_pop", "object_pop"):
            with self.subTest(column=column):
                self.assert_rejected(column)

    def test_pop_bucket_and_z_columns_raise(self):
        for column in ("pop_bucket", "pop_bucket_2", "s_pop_z", "o_pop_log_z"):
            with self.subTest(column=column):
                self.assert_rejected(column)

    def test_risk_score_column_raises(self):
        self.assert_rejected("forgetting_risk")

    def test_unapproved_and_duplicate_columns_raise(self):
        with self.assertRaisesRegex(ValueError, "approved"):
            validate_predictor_columns([*PREDICTOR_COLUMNS, "confidence"])
        with self.assertRaisesRegex(ValueError, "duplicates"):
            validate_predictor_columns(["subject_pop_log", "subject_pop_log"])
        with self.assertRaises(TypeError):
            validate_predictor_columns("subject_pop_log")

    def test_approved_sets_pass(self):
        self.assertEqual(validate_predictor_columns(ENRICHED_PREDICTOR_COLUMNS),
                         ENRICHED_PREDICTOR_COLUMNS)


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.dataset = make_enriched_dataset()

    def assert_invalid(self, frame, pattern):
        with self.assertRaisesRegex(ValueError, pattern):
            validate_enriched_dataset(frame)

    def test_valid_dataset_and_known_population(self):
        validated = validate_enriched_dataset(self.dataset)
        self.assertEqual(len(validated), len(self.dataset))
        known = select_known_population(validated)
        self.assertEqual(len(known), N_KNOWN)
        self.assertTrue((known["guessable"] == 0).all())
        self.assertEqual(fp.TARGET_COLUMN, "forgotten_lr1e5")

    def test_load_from_csv_with_and_without_count_check(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "enriched.csv"
            self.dataset.to_csv(path, index=False)
            self.assertEqual(len(load_enriched_dataset(path, expect_counts=False)), 48)
            with self.assertRaisesRegex(ValueError, "expected 929"):
                load_enriched_dataset(path)

    def test_duplicate_ids_fail(self):
        frame = self.dataset.copy()
        frame.loc[1, "id"] = frame.loc[0, "id"]
        self.assert_invalid(frame, "duplicate")

    def test_missing_required_column_fails(self):
        for column in ("question_chars", TARGET_COLUMN, "guessable", "et_other"):
            with self.subTest(column=column):
                self.assert_invalid(self.dataset.drop(columns=column), "missing required")

    def test_non_binary_target_and_guessable_fail(self):
        for column in (TARGET_COLUMN, "guessable"):
            frame = self.dataset.copy()
            frame.loc[0, column] = 2
            with self.subTest(column=column):
                self.assert_invalid(frame, column)

    def test_non_binary_entity_type_fails(self):
        frame = self.dataset.copy()
        frame["et_place"] = frame["et_place"] * 2
        self.assert_invalid(frame, "et_place")

    def test_inconsistent_onehots_fail(self):
        frame = self.dataset.copy()
        frame.loc[0, ["et_person", "et_place", "et_creative_work", "et_other"]] = [1, 1, 0, 0]
        self.assert_invalid(frame, "et_")
        frame = self.dataset.copy()
        row = frame.index[frame["relation"] == "capital"][0]
        frame.loc[row, ["rel_capital", "rel_country"]] = [0, 1]
        self.assert_invalid(frame, "relation")

    def test_non_finite_predictor_fails(self):
        frame = self.dataset.copy()
        frame["question_length"] = frame["question_length"].astype(float)
        frame.loc[3, "question_length"] = np.inf
        self.assert_invalid(frame, "question_length")

    def test_missing_predictor_value_fails(self):
        frame = self.dataset.copy()
        frame.loc[3, "answer_chars"] = np.nan
        self.assert_invalid(frame, "answer_chars")

    def test_rows_never_silently_dropped(self):
        frame = self.dataset.copy()
        frame["subject_pop_log"] = frame["subject_pop_log"].astype(object)
        frame.loc[5, "subject_pop_log"] = "bad"
        self.assert_invalid(frame, "subject_pop_log")


class FrozenSplitTests(SplitDirMixin, unittest.TestCase):
    def test_mapping_is_by_id_and_disjoint(self):
        result = run_frozen_seed(self.dataset, 0, "enriched", "logistic_regression",
                                 splits_dir=self.splits_dir)
        assignments = self.assignments(0).set_index("id")["split"]
        self.assertTrue((assignments.loc[result.train_ids] == "train").all())
        self.assertTrue((assignments.loc[result.validation_ids] == "val").all())
        self.assertTrue((assignments.loc[result.test_ids] == "test").all())
        partitions = [set(result.train_ids), set(result.validation_ids), set(result.test_ids)]
        self.assertFalse(partitions[0] & partitions[1] or partitions[0] & partitions[2]
                         or partitions[1] & partitions[2])
        known_ids = set(self.dataset.loc[self.dataset.guessable == 0, "id"])
        self.assertEqual(set().union(*partitions), known_ids)
        self.assertEqual((len(result.train_ids), len(result.validation_ids),
                          len(result.test_ids)), (28, 6, 6))

    def test_split_files_are_not_rewritten(self):
        before = {p: p.read_bytes() for p in (self.splits_dir / "known").glob("*.csv")}
        for seed in (0, 1, 2):
            run_frozen_seed(self.dataset, seed, "original", "logistic_regression",
                            splits_dir=self.splits_dir)
        after = {p: p.read_bytes() for p in (self.splits_dir / "known").glob("*.csv")}
        self.assertEqual(before, after)

    def test_missing_split_file_raises_instead_of_regenerating(self):
        (self.splits_dir / "known" / "seed1.csv").unlink()
        with self.assertRaises(ValueError):
            run_frozen_seed(self.dataset, 1, "enriched", "lightgbm", splits_dir=self.splits_dir)
        self.assertFalse((self.splits_dir / "known" / "seed1.csv").exists())

    def test_split_missing_known_id_raises(self):
        path = self.splits_dir / "known" / "seed0.csv"
        pd.read_csv(path).iloc[1:].to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "missing"):
            run_frozen_seed(self.dataset, 0, "enriched", "lightgbm", splits_dir=self.splits_dir)


class RecordingModel:
    """Spy that records the rows it is fitted on and scores on."""

    def __init__(self, seed):
        self.seed = seed
        self.fit_frames = []
        self.predict_frames = []

    def fit(self, features, labels):
        self.fit_frames.append(features.copy())
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, features):
        self.predict_frames.append(features.copy())
        positive = np.linspace(0.1, 0.9, len(features))
        return np.column_stack([1 - positive, positive])


class OneSeedRunnerTests(SplitDirMixin, unittest.TestCase):
    def test_fits_on_train_rows_and_scores_test_rows_only(self):
        spies = []

        def factory(seed):
            spies.append(RecordingModel(seed))
            return spies[-1]

        known = select_known_population(self.dataset)
        result = run_frozen_seed(self.dataset, 2, "enriched", "lightgbm",
                                 splits_dir=self.splits_dir,
                                 model_factories={"lightgbm": factory})
        spy = spies[0]
        self.assertEqual(spy.seed, 2)
        self.assertEqual(len(spy.fit_frames), 1)
        self.assertEqual(len(spy.predict_frames), 1)
        fitted_ids = known.loc[spy.fit_frames[0].index, "id"].to_numpy()
        scored_ids = known.loc[spy.predict_frames[0].index, "id"].to_numpy()
        np.testing.assert_array_equal(fitted_ids, result.train_ids)
        np.testing.assert_array_equal(scored_ids, result.test_ids)
        self.assertEqual(list(spy.fit_frames[0].columns), list(ENRICHED_PREDICTOR_COLUMNS))

    def test_real_models_return_valid_test_scores_and_metrics(self):
        for model_name in ("logistic_regression", "lightgbm"):
            for feature_set, n_features in (("original", 18), ("enriched", 26)):
                with self.subTest(model=model_name, feature_set=feature_set):
                    result = run_frozen_seed(self.dataset, 0, feature_set, model_name,
                                             splits_dir=self.splits_dir)
                    self.assertEqual(result.predictor_names, predictor_columns(feature_set))
                    self.assertEqual(len(result.predictor_names), n_features)
                    self.assertEqual(result.test_scores.shape, (len(result.test_ids),))
                    self.assertTrue(np.isfinite(result.test_scores).all())
                    self.assertTrue(((result.test_scores >= 0) & (result.test_scores <= 1)).all())
                    self.assertEqual(set(result.metrics), set(METRIC_NAMES))
                    self.assertEqual(result.feature_set, feature_set)
                    self.assertEqual(result.model_name, model_name)
                    self.assertEqual(result.seed, 0)
                    self.assertEqual(getattr(result.model, "n_features_in_", None), n_features)

    def test_test_labels_match_target_by_id(self):
        result = run_frozen_seed(self.dataset, 1, "enriched", "logistic_regression",
                                 splits_dir=self.splits_dir)
        by_id = self.dataset.set_index("id")[TARGET_COLUMN]
        np.testing.assert_array_equal(result.test_labels, by_id.loc[result.test_ids].to_numpy())

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            run_frozen_seed(self.dataset, 3, "enriched", "lightgbm", splits_dir=self.splits_dir)
        with self.assertRaises(ValueError):
            run_frozen_seed(self.dataset, 0, "enriched", "svm", splits_dir=self.splits_dir)
        with self.assertRaises(ValueError):
            run_frozen_seed(self.dataset, 0, "all", "lightgbm", splits_dir=self.splits_dir)


class ModelReuseTests(unittest.TestCase):
    def test_factories_are_task5_factories(self):
        self.assertIs(MODEL_FACTORIES["logistic_regression"], make_logistic_pipeline)
        self.assertIs(MODEL_FACTORIES["lightgbm"], make_lightgbm_classifier)
        self.assertEqual(set(MODEL_FACTORIES), {"logistic_regression", "lightgbm"})

    def test_task5_hyperparameters_unchanged(self):
        lgbm = MODEL_FACTORIES["lightgbm"](1).get_params()
        for key, value in dict(n_estimators=100, learning_rate=0.05, max_depth=4,
                               num_leaves=15, class_weight="balanced", random_state=1,
                               n_jobs=1).items():
            self.assertEqual(lgbm[key], value)
        logistic = MODEL_FACTORIES["logistic_regression"](1)
        self.assertEqual([name for name, _ in logistic.steps], ["scaler", "classifier"])
        classifier = logistic.named_steps["classifier"].get_params()
        self.assertEqual((classifier["class_weight"], classifier["max_iter"],
                          classifier["random_state"]), ("balanced", 1000, 1))


def make_task5_frame():
    rows = []
    for population in ("known", "all"):
        for seed in (0, 1, 2):
            for model in ("logistic_regression_full", "lightgbm_full",
                          "logistic_regression_popularity_only", "prevalence"):
                rows.append({
                    "population": population, "seed": seed, "model": model,
                    "n_train": 199, "n_val": 43, "n_test": 43, "train_prevalence": 0.54,
                    **{metric: 0.1 * (seed + 1) + 0.01 * len(model) + 0.001 * i
                       for i, metric in enumerate(METRIC_NAMES)},
                })
    return pd.DataFrame(rows)


def matching_rerun(baseline):
    rerun = baseline.copy()
    rerun.insert(4, "n_features", 18)
    return rerun.loc[:, METRIC_RECORD_COLUMNS]


class BaselineReproductionTests(unittest.TestCase):
    def test_selects_known_lr_and_lightgbm_in_task17_naming(self):
        baseline = task5_baseline_from_frame(make_task5_frame())
        self.assertEqual(len(baseline), 6)
        self.assertEqual(set(baseline.population), {"known"})
        self.assertEqual(list(baseline.model.unique()), ["logistic_regression", "lightgbm"])
        self.assertEqual(set(baseline.feature_set), {"original"})
        self.assertEqual(list(baseline.seed), [0, 1, 2, 0, 1, 2])
        self.assertIn("n_validation", baseline.columns)

    def test_missing_seed_raises(self):
        frame = make_task5_frame()
        frame = frame[~((frame.model == "lightgbm_full") & (frame.seed == 2))]
        with self.assertRaisesRegex(ValueError, "must contain"):
            task5_baseline_from_frame(frame)

    def test_matching_metrics_pass(self):
        baseline = task5_baseline_from_frame(make_task5_frame())
        rerun = matching_rerun(baseline)
        rerun["auprc"] = rerun["auprc"] + 1e-15
        compare_with_task5_baseline(rerun, baseline)

    def test_mismatch_raises_useful_error(self):
        baseline = task5_baseline_from_frame(make_task5_frame())
        rerun = matching_rerun(baseline)
        rerun.loc[(rerun.model == "lightgbm") & (rerun.seed == 1), "f1"] += 1e-6
        with self.assertRaises(ValueError) as context:
            compare_with_task5_baseline(rerun, baseline)
        message = str(context.exception)
        self.assertIn("lightgbm seed 1 f1", message)
        self.assertIn("task5=", message)
        self.assertNotIn("auprc", message)

    def test_count_mismatch_and_missing_baseline_row_raise(self):
        baseline = task5_baseline_from_frame(make_task5_frame())
        rerun = matching_rerun(baseline)
        rerun.loc[0, "n_test"] = 42
        with self.assertRaisesRegex(ValueError, "n_test"):
            compare_with_task5_baseline(rerun, baseline)
        with self.assertRaisesRegex(ValueError, "no committed Task 5 row"):
            compare_with_task5_baseline(matching_rerun(baseline), baseline.iloc[1:])

    def test_enriched_rows_cannot_be_compared(self):
        baseline = task5_baseline_from_frame(make_task5_frame())
        rerun = matching_rerun(baseline)
        rerun["feature_set"] = "enriched"
        with self.assertRaisesRegex(ValueError, "original"):
            compare_with_task5_baseline(rerun, baseline)

    def test_committed_task5_file_loads_read_only(self):
        path = fp.TASK5_METRICS_PATH
        if not path.exists():
            self.skipTest("committed Task 5 metrics not present")
        before = path.read_bytes()
        baseline = fp.load_task5_baseline(path)
        self.assertEqual(len(baseline), 6)
        self.assertEqual(path.read_bytes(), before)


class MetricRecordSchemaTests(SplitDirMixin, unittest.TestCase):
    def test_record_columns_match_approved_schema(self):
        self.assertEqual(METRIC_RECORD_COLUMNS, (
            "population", "model", "feature_set", "seed", "n_features", "n_train",
            "n_validation", "n_test", "auprc", "f1", "recall_at_1", "recall_at_5",
            "recall_at_10",
        ))
        result = run_frozen_seed(self.dataset, 0, "enriched", "lightgbm",
                                 splits_dir=self.splits_dir)
        record = metric_record(result)
        self.assertEqual(tuple(record), METRIC_RECORD_COLUMNS)
        self.assertEqual((record["population"], record["n_features"], record["n_train"],
                          record["n_validation"], record["n_test"]), ("known", 26, 28, 6, 6))
        frame = metric_records_frame([result])
        self.assertEqual(tuple(frame.columns), METRIC_RECORD_COLUMNS)


# --------------------------------------------------------------------------
# Commit 2: feature importance
# --------------------------------------------------------------------------

from sklearn.base import BaseEstimator, ClassifierMixin  # noqa: E402

from src.final_predictors import (  # noqa: E402
    FEATURE_GROUPS,
    IMPORTANCE_COLUMNS,
    IMPORTANCE_SUMMARY_COLUMNS,
    aggregate_importance,
    extract_lgbm_gain,
    extract_lr_coefficients,
    feature_permutation_importance,
    grouped_permutation_importance,
    plot_grouped_permutation_importance,
    validate_feature_groups,
)


def signal_frame(n=300, seed=0):
    """'pos' raises risk, 'neg' lowers it, 'noise' is unrelated, 'const' is constant."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "pos": rng.normal(size=n),
        "neg": rng.normal(size=n),
        "noise": rng.normal(size=n),
        "const": np.zeros(n),
    })
    logits = 2.0 * frame["pos"] - 2.0 * frame["neg"]
    labels = (rng.uniform(size=n) < 1 / (1 + np.exp(-logits))).astype(int)
    return frame, labels


class SpyClassifier(ClassifierMixin, BaseEstimator):
    """Sklearn-compatible scorer-only model: risk = sigmoid(direction * column)."""

    def __init__(self, column="pos", direction=1.0):
        self.column = column
        self.direction = direction

    def fit(self, features, labels):
        self.classes_ = np.array([0, 1])
        self.seen_ = []
        return self

    def predict_proba(self, features):
        self.seen_.append(features.copy())
        positive = 1 / (1 + np.exp(-self.direction * features[self.column].to_numpy(float)))
        return np.column_stack([1 - positive, positive])


class LogisticCoefficientTests(unittest.TestCase):
    def setUp(self):
        self.features, self.labels = signal_frame()
        self.names = tuple(self.features.columns)
        self.pipeline = make_logistic_pipeline(0).fit(self.features, self.labels)

    def test_names_signs_abs_and_constant(self):
        rows = extract_lr_coefficients(self.pipeline, self.names, seed=4)
        self.assertEqual(tuple(rows.columns), IMPORTANCE_COLUMNS)
        self.assertEqual(tuple(rows.feature), self.names)
        self.assertEqual(set(rows.model), {"logistic_regression"})
        self.assertEqual(set(rows.importance_type), {"lr_coefficient"})
        self.assertEqual(set(rows.level), {"feature"})
        self.assertEqual(set(rows.seed), {4})
        values = rows.set_index("feature")["importance"]
        self.assertGreater(values["pos"], 0.5)
        self.assertLess(values["neg"], -0.5)
        self.assertAlmostEqual(values["const"], 0.0, places=8)
        np.testing.assert_allclose(rows.abs_importance, rows.importance.abs())
        np.testing.assert_allclose(
            rows.importance, self.pipeline.named_steps["classifier"].coef_[0])

    def test_feature_count_and_order_mismatch_raise(self):
        with self.assertRaisesRegex(ValueError, "coefficients"):
            extract_lr_coefficients(self.pipeline, self.names[:3], seed=0)
        with self.assertRaisesRegex(ValueError, "do not match"):
            extract_lr_coefficients(self.pipeline, self.names[::-1], seed=0)

    def test_unfitted_or_wrong_type_raise(self):
        with self.assertRaisesRegex(ValueError, "not fitted"):
            extract_lr_coefficients(make_logistic_pipeline(0), self.names, seed=0)
        with self.assertRaises(TypeError):
            extract_lr_coefficients(make_lightgbm_classifier(0), self.names, seed=0)


class LightGBMGainTests(unittest.TestCase):
    def setUp(self):
        self.features, self.labels = signal_frame(n=400)
        self.names = tuple(self.features.columns)
        self.model = make_lightgbm_classifier(0).fit(self.features, self.labels)

    def test_gain_normalized_with_zero_gain_feature_kept(self):
        rows = extract_lgbm_gain(self.model, self.names, seed=1)
        self.assertEqual(tuple(rows.columns), IMPORTANCE_COLUMNS)
        self.assertEqual(tuple(rows.feature), self.names)
        self.assertEqual(set(rows.importance_type), {"lgbm_gain_normalized"})
        self.assertAlmostEqual(rows.importance.sum(), 1.0, places=12)
        values = rows.set_index("feature")["importance"]
        self.assertEqual(values["const"], 0.0)
        self.assertGreater(values["pos"], values["noise"])

    def test_uses_gain_not_split_count(self):
        rows = extract_lgbm_gain(self.model, self.names, seed=0)
        booster = self.model.booster_
        gain = booster.feature_importance(importance_type="gain")
        split = booster.feature_importance(importance_type="split")
        np.testing.assert_allclose(rows.importance, gain / gain.sum())
        self.assertFalse(np.allclose(rows.importance, split / split.sum()))
        self.assertEqual(self.model.importance_type, "split")  # why feature_importances_ is avoided

    def test_zero_total_gain_raises(self):
        class ZeroBooster:
            def feature_name(self):
                return ["pos", "neg"]

            def feature_importance(self, importance_type="split"):
                return np.zeros(2)

        class Fitted:
            booster_ = ZeroBooster()

        with self.assertRaisesRegex(ValueError, "total gain is zero"):
            extract_lgbm_gain(Fitted(), ("pos", "neg"), seed=0)

    def test_feature_name_mismatch_and_unfitted_raise(self):
        with self.assertRaisesRegex(ValueError, "do not match"):
            extract_lgbm_gain(self.model, self.names[::-1], seed=0)
        with self.assertRaisesRegex(ValueError, "not fitted"):
            extract_lgbm_gain(make_lightgbm_classifier(0), self.names, seed=0)


class FeaturePermutationTests(unittest.TestCase):
    def setUp(self):
        features, labels = signal_frame(n=900, seed=3)
        self.train_x, self.train_y = features.iloc[:500], labels[:500]
        self.test_x, self.test_y = features.iloc[500:], labels[500:]
        self.model = make_logistic_pipeline(0).fit(self.train_x, self.train_y)

    def test_one_finite_row_per_feature_and_deterministic(self):
        first = feature_permutation_importance(self.model, "logistic_regression",
                                               self.test_x, self.test_y, seed=0, n_repeats=10)
        second = feature_permutation_importance(self.model, "logistic_regression",
                                                self.test_x, self.test_y, seed=0, n_repeats=10)
        self.assertEqual(tuple(first.columns), IMPORTANCE_COLUMNS)
        self.assertEqual(tuple(first.feature), tuple(self.test_x.columns))
        self.assertEqual(set(first.level), {"feature"})
        self.assertEqual(set(first.importance_type), {"permutation_auprc"})
        self.assertTrue(np.isfinite(first[["importance", "importance_std_repeats"]]).all().all())
        pd.testing.assert_frame_equal(first, second)

    def test_noise_feature_unimportant_and_signal_important(self):
        rows = feature_permutation_importance(self.model, "logistic_regression",
                                              self.test_x, self.test_y, seed=0, n_repeats=20)
        values = rows.set_index("feature")["importance"]
        self.assertGreater(values["pos"], 0.05)
        self.assertGreater(values["neg"], 0.05)
        self.assertLess(abs(values["noise"]), 0.02)
        self.assertEqual(values["const"], 0.0)

    def test_scores_only_supplied_rows(self):
        spy = SpyClassifier().fit(self.train_x, self.train_y)
        feature_permutation_importance(spy, "spy", self.test_x.iloc[:50], self.test_y[:50],
                                       seed=0, n_repeats=3)
        self.assertTrue(spy.seen_)
        self.assertEqual({len(frame) for frame in spy.seen_}, {50})

    def test_negative_importance_is_not_clipped(self):
        reversed_model = SpyClassifier(column="pos", direction=-1.0).fit(self.test_x, self.test_y)
        rows = feature_permutation_importance(reversed_model, "spy", self.test_x, self.test_y,
                                              seed=0, n_repeats=10)
        self.assertLess(rows.set_index("feature").loc["pos", "importance"], 0)

    def test_invalid_heldout_inputs_raise(self):
        with self.assertRaisesRegex(ValueError, "matching lengths"):
            feature_permutation_importance(self.model, "m", self.test_x, self.test_y[:-1], 0)
        with self.assertRaisesRegex(ValueError, "both classes"):
            feature_permutation_importance(self.model, "m", self.test_x,
                                           np.zeros(len(self.test_x), dtype=int), 0)
        with self.assertRaisesRegex(ValueError, "binary"):
            feature_permutation_importance(self.model, "m", self.test_x,
                                           np.full(len(self.test_x), 2), 0)
        with self.assertRaisesRegex(ValueError, "empty"):
            feature_permutation_importance(self.model, "m", self.test_x.iloc[:0],
                                           self.test_y[:0], 0)


class GroupedPermutationTests(unittest.TestCase):
    def test_default_groups_cover_all_26_predictors_once(self):
        validate_feature_groups()
        self.assertEqual(list(FEATURE_GROUPS), [
            "subject_popularity", "object_popularity", "relation", "entity_type",
            "question_length", "answer_length",
        ])
        members = [c for columns in FEATURE_GROUPS.values() for c in columns]
        self.assertEqual(sorted(members), sorted(ENRICHED_PREDICTOR_COLUMNS))
        self.assertEqual(len(members), 26)
        self.assertEqual(FEATURE_GROUPS["relation"], tuple(RELATION_FEATURES))
        self.assertEqual(len(FEATURE_GROUPS["relation"]), 16)

    def test_invalid_groups_fail(self):
        groups = dict(FEATURE_GROUPS)
        with self.assertRaisesRegex(ValueError, "more than one group"):
            validate_feature_groups({**groups, "extra": ("subject_pop_log",)})
        with self.assertRaisesRegex(ValueError, "not approved"):
            validate_feature_groups({**groups, "extra": ("confidence",)})
        missing = {k: v for k, v in groups.items() if k != "answer_length"}
        with self.assertRaisesRegex(ValueError, "not covered"):
            validate_feature_groups(missing)

    def test_same_row_permutation_applied_to_whole_group(self):
        n = 60
        rng = np.random.default_rng(0)
        features = pd.DataFrame({"a": np.arange(n, dtype=float),
                                 "b": np.arange(n, dtype=float) + 1000.0,
                                 "c": rng.normal(size=n)})
        labels = (features["a"] > n / 2).astype(int).to_numpy()
        spy = SpyClassifier(column="a").fit(features, labels)
        grouped_permutation_importance(spy, "spy", features, labels, seed=0,
                                       groups={"ab": ("a", "b"), "c": ("c",)}, n_repeats=5)
        permuted_ab = [f for f in spy.seen_ if not f["a"].equals(features["a"])]
        self.assertEqual(len(permuted_ab), 5)
        for frame in permuted_ab:
            np.testing.assert_array_equal(frame["b"] - frame["a"], 1000.0)
            pd.testing.assert_series_equal(frame["c"], features["c"])
        self.assertEqual({len(f) for f in spy.seen_}, {n})

    def test_six_rows_per_model_seed_and_deterministic(self):
        dataset = make_enriched_dataset(seed=1)
        known = select_known_population(validate_enriched_dataset(dataset))
        features = select_enriched_predictors(known)
        labels = known[TARGET_COLUMN].to_numpy()
        for model_name, factory in MODEL_FACTORIES.items():
            model = factory(0).fit(features, labels)
            with self.subTest(model=model_name):
                first = grouped_permutation_importance(model, model_name, features, labels,
                                                       seed=2, n_repeats=5)
                again = grouped_permutation_importance(model, model_name, features, labels,
                                                       seed=2, n_repeats=5)
                self.assertEqual(tuple(first.columns), IMPORTANCE_COLUMNS)
                self.assertEqual(list(first.feature), list(FEATURE_GROUPS))
                self.assertEqual(set(first.level), {"group"})
                self.assertEqual(set(first.seed), {2})
                self.assertTrue(np.isfinite(first.importance).all())
                pd.testing.assert_frame_equal(first, again)

    def test_columns_must_match_groups(self):
        features, labels = signal_frame(n=50)
        model = SpyClassifier().fit(features, labels)
        with self.assertRaisesRegex(ValueError, "not covered"):
            grouped_permutation_importance(model, "spy", features, labels, seed=0,
                                           groups={"p": ("pos",)}, n_repeats=2)


def importance_rows(model, importance_type, level, values_by_feature):
    rows = []
    for feature, values in values_by_feature.items():
        for seed, value in enumerate(values):
            rows.append({"model": model, "importance_type": importance_type, "level": level,
                         "feature": feature, "seed": seed, "importance": value,
                         "importance_std_repeats": np.nan,
                         "abs_importance": abs(value) if importance_type == "lr_coefficient"
                         else np.nan})
    return rows


class AggregationTests(unittest.TestCase):
    def setUp(self):
        rows = importance_rows("logistic_regression", "lr_coefficient", "feature", {
            "a": [1.0, 2.0, 3.0],
            "b": [-4.0, -1.0, 2.0],
            "z": [0.0, 0.0, 0.0],
        })
        rows += importance_rows("lightgbm", "lgbm_gain_normalized", "feature", {
            "a": [0.5, 0.5, 0.5], "b": [0.5, 0.5, 0.5], "z": [0.0, 0.0, 0.0],
        })
        rows += importance_rows("lightgbm", "permutation_auprc", "group", {
            "g1": [0.1, 0.2, 0.3], "g2": [-0.05, 0.0, 0.02],
        })
        self.summary = aggregate_importance(pd.DataFrame(rows, columns=IMPORTANCE_COLUMNS))
        self.indexed = self.summary.set_index(["model", "importance_type", "feature"])

    def row(self, model, importance_type, feature):
        return self.indexed.loc[(model, importance_type, feature)]

    def test_schema(self):
        self.assertEqual(tuple(self.summary.columns), IMPORTANCE_SUMMARY_COLUMNS)
        self.assertEqual(len(self.summary), 8)

    def test_mean_sd_and_abs(self):
        a = self.row("logistic_regression", "lr_coefficient", "a")
        self.assertAlmostEqual(a.mean_importance, 2.0)
        self.assertAlmostEqual(a.std_importance, 1.0)  # ddof=1
        b = self.row("logistic_regression", "lr_coefficient", "b")
        self.assertAlmostEqual(b.mean_importance, -1.0)
        self.assertAlmostEqual(b.std_importance, np.std([-4.0, -1.0, 2.0], ddof=1))
        self.assertAlmostEqual(b.mean_abs_importance, 7 / 3)
        self.assertTrue(np.isnan(self.row("lightgbm", "lgbm_gain_normalized", "a")
                                 .mean_abs_importance))

    def test_sign_consistency(self):
        self.assertEqual(self.row("logistic_regression", "lr_coefficient", "a").sign_consistency, 3)
        self.assertEqual(self.row("logistic_regression", "lr_coefficient", "b").sign_consistency, 2)
        self.assertEqual(self.row("logistic_regression", "lr_coefficient", "z").sign_consistency, 0)
        self.assertTrue(np.isnan(self.row("lightgbm", "permutation_auprc", "g1").sign_consistency))

    def test_nonzero_counts(self):
        self.assertEqual(self.row("logistic_regression", "lr_coefficient", "z").n_seeds_nonzero, 0)
        self.assertEqual(self.row("lightgbm", "permutation_auprc", "g2").n_seeds_nonzero, 2)
        self.assertEqual(self.row("lightgbm", "permutation_auprc", "g1").n_seeds, 3)

    def test_rank_direction_and_ties(self):
        # LR ranks by mean |coef|: b (7/3) > a (2) > z (0).
        self.assertEqual(self.row("logistic_regression", "lr_coefficient", "b")["rank"], 1)
        self.assertEqual(self.row("logistic_regression", "lr_coefficient", "a")["rank"], 2)
        self.assertEqual(self.row("logistic_regression", "lr_coefficient", "z")["rank"], 3)
        # Gain ties use method="min".
        self.assertEqual(self.row("lightgbm", "lgbm_gain_normalized", "a")["rank"], 1)
        self.assertEqual(self.row("lightgbm", "lgbm_gain_normalized", "b")["rank"], 1)
        self.assertEqual(self.row("lightgbm", "lgbm_gain_normalized", "z")["rank"], 3)
        # Permutation ranks by signed mean, descending, within its own type/level.
        self.assertEqual(self.row("lightgbm", "permutation_auprc", "g1")["rank"], 1)
        self.assertEqual(self.row("lightgbm", "permutation_auprc", "g2")["rank"], 2)

    def test_duplicate_rows_raise(self):
        rows = importance_rows("lightgbm", "permutation_auprc", "group", {"g": [0.1, 0.2]})
        frame = pd.DataFrame(rows + rows[:1], columns=IMPORTANCE_COLUMNS)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            aggregate_importance(frame)


def grouped_figure_rows():
    rows = []
    for model_index, model in enumerate(("logistic_regression", "lightgbm")):
        values = {group: [0.01 * (i + 1) + 0.005 * seed - 0.002 * model_index
                          for seed in range(3)]
                  for i, group in enumerate(FEATURE_GROUPS)}
        rows += importance_rows(model, "permutation_auprc", "group", values)
    return pd.DataFrame(rows, columns=IMPORTANCE_COLUMNS)


class FigureTests(unittest.TestCase):
    def test_two_panel_png_written_to_supplied_path_only(self):
        real_existed = fp.TASK17_FIGURE_OUTPUT.exists()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "importance.png"
            figure = plot_grouped_permutation_importance(grouped_figure_rows(), path)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(len(figure.axes), 2)
        self.assertEqual([axis.get_title() for axis in figure.axes],
                         ["Logistic Regression", "LightGBM"])
        labels = [tick.get_text() for tick in figure.axes[0].get_yticklabels()]
        self.assertEqual(len(labels), 6)
        self.assertIn("Relation (16 one-hot)", labels)
        self.assertEqual(len(figure.axes[0].patches), 6)  # one bar per group
        self.assertEqual(fp.TASK17_FIGURE_OUTPUT.exists(), real_existed)

    def test_missing_model_or_group_raises(self):
        rows = grouped_figure_rows()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.png"
            with self.assertRaisesRegex(ValueError, "lightgbm"):
                plot_grouped_permutation_importance(rows[rows.model != "lightgbm"], path)
            with self.assertRaisesRegex(ValueError, "groups"):
                plot_grouped_permutation_importance(rows[rows.feature != "relation"], path)
            with self.assertRaisesRegex(ValueError, ".png"):
                plot_grouped_permutation_importance(rows, Path(directory) / "x.pdf")


class NoOutputFilesTests(SplitDirMixin, unittest.TestCase):
    def test_runner_writes_no_files(self):
        with patch.object(pd.DataFrame, "to_csv", side_effect=AssertionError("wrote CSV")):
            run_frozen_seed(self.dataset, 0, "enriched", "logistic_regression",
                            splits_dir=self.splits_dir)


if __name__ == "__main__":
    unittest.main()
