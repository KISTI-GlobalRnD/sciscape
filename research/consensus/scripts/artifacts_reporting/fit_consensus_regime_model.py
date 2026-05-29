"""Fit simple predictive models for when consensus wins a local review case."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any
import sys

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
SCRIPT_ROOT = REPO_ROOT / "research/consensus/scripts"
_SCRIPT_PATHS = [REPO_ROOT, SCRIPT_ROOT]
_SCRIPT_PATHS.extend(path for path in SCRIPT_ROOT.rglob("*") if path.is_dir())
for _script_path in reversed(_SCRIPT_PATHS):
    _script_path_str = str(_script_path)
    if _script_path_str not in sys.path:
        sys.path.insert(0, _script_path_str)


import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from _common import save_json

def _consensus_side(payload: dict[str, Any]) -> tuple[str, str, str]:
    label_a = payload["label_a"]
    label_b = payload["label_b"]
    if "consensus" in label_a:
        return ("A", label_a, label_b)
    if "consensus" in label_b:
        return ("B", label_b, label_a)
    raise ValueError(f"Review does not contain a consensus label: {label_a}, {label_b}")

def _field_prefix(field_name: str) -> str:
    parts = field_name.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else field_name

def _build_rows(review_paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in review_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        consensus_side, consensus_label, baseline_label = _consensus_side(payload)
        for case in payload.get("reviewed_cases", []):
            winner = case["comparison"]["winner"]
            if winner not in {"A", "B"}:
                continue
            consensus_wins = int(winner == consensus_side)
            consensus_cluster_size = case["method_a_cluster_size"] if consensus_side == "A" else case["method_b_cluster_size"]
            baseline_cluster_size = case["method_b_cluster_size"] if consensus_side == "A" else case["method_a_cluster_size"]
            cluster_ratio = (
                float(consensus_cluster_size) / float(baseline_cluster_size)
                if baseline_cluster_size
                else math.nan
            )
            rows.append(
                {
                    "review_json": str(path),
                    "field": payload["field"],
                    "field_prefix": _field_prefix(payload["field"]),
                    "top_k": int(payload["top_k"]),
                    "baseline_label": baseline_label,
                    "consensus_label": consensus_label,
                    "consensus_wins": consensus_wins,
                    "rank_jaccard": float(case["rank_jaccard"]),
                    "overlap_size": int(case["overlap_size"]),
                    "mean_abs_rank_shift": float(case["mean_abs_rank_shift"]),
                    "max_abs_rank_shift": int(case["max_abs_rank_shift"]),
                    "shift_score": float(case["shift_score"]),
                    "cluster_overlap_coeff": float(case.get("cluster_overlap_coeff", math.nan)),
                    "cluster_changed": int(bool(case["cluster_changed"])),
                    "log_consensus_cluster_size": math.log1p(float(consensus_cluster_size)),
                    "log_baseline_cluster_size": math.log1p(float(baseline_cluster_size)),
                    "cluster_size_ratio": cluster_ratio,
                }
            )
    return rows

def _feature_matrix(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, np.ndarray]:
    X = pd.DataFrame(
        [
        {
            "field_prefix": row["field_prefix"],
            "baseline_label": row["baseline_label"],
            "top_k": row["top_k"],
            "rank_jaccard": row["rank_jaccard"],
            "overlap_size": row["overlap_size"],
            "mean_abs_rank_shift": row["mean_abs_rank_shift"],
            "max_abs_rank_shift": row["max_abs_rank_shift"],
            "shift_score": row["shift_score"],
            "cluster_overlap_coeff": row["cluster_overlap_coeff"],
            "cluster_changed": row["cluster_changed"],
            "log_consensus_cluster_size": row["log_consensus_cluster_size"],
            "log_baseline_cluster_size": row["log_baseline_cluster_size"],
            "cluster_size_ratio": row["cluster_size_ratio"],
        }
        for row in rows
        ]
    )
    y = np.asarray([row["consensus_wins"] for row in rows], dtype=int)
    return X, y

def _make_logistic_pipeline() -> Pipeline:
    numeric_features = [
        "top_k",
        "rank_jaccard",
        "overlap_size",
        "mean_abs_rank_shift",
        "max_abs_rank_shift",
        "shift_score",
        "cluster_overlap_coeff",
        "cluster_changed",
        "log_consensus_cluster_size",
        "log_baseline_cluster_size",
        "cluster_size_ratio",
    ]
    categorical_features = ["field_prefix", "baseline_label"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )

def _fit_tree(rows: pd.DataFrame, y: np.ndarray) -> dict[str, Any]:
    numeric_features = [
        "top_k",
        "rank_jaccard",
        "overlap_size",
        "mean_abs_rank_shift",
        "max_abs_rank_shift",
        "shift_score",
        "cluster_overlap_coeff",
        "cluster_changed",
        "log_consensus_cluster_size",
        "log_baseline_cluster_size",
        "cluster_size_ratio",
    ]
    categorical_features = ["field_prefix", "baseline_label"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    X = preprocessor.fit_transform(rows)
    feature_names = list(preprocessor.get_feature_names_out())
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42, class_weight="balanced")
    tree.fit(X, y)
    importances = sorted(
        (
            {"feature": name, "importance": round(float(importance), 6)}
            for name, importance in zip(feature_names, tree.feature_importances_)
            if importance > 0
        ),
        key=lambda item: (-item["importance"], item["feature"]),
    )
    return {
        "feature_importances": importances,
        "rules": export_text(tree, feature_names=feature_names),
    }

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_json", nargs="+", type=Path, help="One or more *_rank_shift_review.json files")
    parser.add_argument("-o", "--output", type=Path, default=Path("research/consensus/results/taxonomy"))
    parser.add_argument("--stem", type=str, default="consensus_regime_model")
    args = parser.parse_args()

    rows = _build_rows(args.review_json)
    X, y = _feature_matrix(rows)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipeline = _make_logistic_pipeline()
    probas = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
    preds = (probas >= 0.5).astype(int)

    pipeline.fit(X, y)
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = list(preprocessor.get_feature_names_out())
    coefficients = sorted(
        (
            {"feature": feature, "coefficient": round(float(coef), 6)}
            for feature, coef in zip(feature_names, model.coef_[0])
        ),
        key=lambda item: (-abs(item["coefficient"]), item["feature"]),
    )

    summary = {
        "n_cases": len(rows),
        "class_balance": {
            "consensus_wins": int(y.sum()),
            "baseline_wins": int(len(y) - y.sum()),
        },
        "cross_validated_metrics": {
            "accuracy": round(float(accuracy_score(y, preds)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y, preds)), 4),
            "roc_auc": round(float(roc_auc_score(y, probas)), 4),
            "brier": round(float(brier_score_loss(y, probas)), 4),
        },
        "top_positive_coefficients": [row for row in coefficients if row["coefficient"] > 0][:12],
        "top_negative_coefficients": [row for row in coefficients if row["coefficient"] < 0][:12],
        "decision_tree": _fit_tree(X, y),
    }

    out_path = args.output / f"{args.stem}.json"
    save_json(summary, out_path)
    print(f"Saved → {out_path}")

if __name__ == "__main__":
    main()
