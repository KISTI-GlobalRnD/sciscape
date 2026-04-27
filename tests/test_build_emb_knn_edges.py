import importlib.util
from pathlib import Path

import numpy as np
import polars as pl
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_emb_knn_edges.py"
SPEC = importlib.util.spec_from_file_location("build_emb_knn_edges", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_filter_embeddings_by_text_quality_keeps_only_long_enough_rows(tmp_path):
    works_text = tmp_path / "works_text.parquet"
    pl.DataFrame({
        "work_id": ["W1", "W2", "W3"],
        "title": ["Good title", "S", "Another"],
        "abstract": [
            "This abstract is comfortably long.",
            "tiny",
            "",
        ],
    }).write_parquet(works_text)

    emb = np.arange(12, dtype=np.float32).reshape(3, 4)
    work_ids = ["W1", "W2", "W3"]

    filtered_emb, filtered_ids, stats = MODULE.filter_embeddings_by_text_quality(
        emb,
        work_ids,
        field_id=99,
        works_text_path=works_text,
        min_text_len=20,
        min_title_len=3,
        min_abstract_len=10,
        require_abstract=True,
        min_metadata_match=1.0,
    )

    assert filtered_ids == ["W1"]
    assert filtered_emb.shape == (1, 4)
    assert stats["matched_nodes"] == 3
    assert stats["kept_nodes"] == 1
    assert stats["dropped_nodes"] == 2


def test_filter_embeddings_by_text_quality_fails_on_low_metadata_match(tmp_path):
    works_text = tmp_path / "works_text.parquet"
    pl.DataFrame({
        "work_id": ["W1"],
        "title": ["Good title"],
        "abstract": ["This abstract is comfortably long."],
    }).write_parquet(works_text)

    emb = np.arange(12, dtype=np.float32).reshape(3, 4)
    work_ids = ["W1", "W2", "W3"]

    with pytest.raises(ValueError, match="Metadata join rate too low"):
        MODULE.filter_embeddings_by_text_quality(
            emb,
            work_ids,
            field_id=99,
            works_text_path=works_text,
            min_text_len=10,
            min_metadata_match=0.9,
        )


def test_filter_embeddings_keeps_unmatched_metadata_by_default_when_match_threshold_allows(tmp_path):
    works_text = tmp_path / "works_text.parquet"
    pl.DataFrame({
        "work_id": ["W1"],
        "title": ["Good title"],
        "abstract": ["This abstract is comfortably long."],
    }).write_parquet(works_text)

    emb = np.arange(12, dtype=np.float32).reshape(3, 4)
    work_ids = ["W1", "W2", "W3"]

    filtered_emb, filtered_ids, stats = MODULE.filter_embeddings_by_text_quality(
        emb,
        work_ids,
        field_id=99,
        works_text_path=works_text,
        min_text_len=10,
        min_metadata_match=0.3,
    )

    assert filtered_ids == ["W1", "W2", "W3"]
    assert filtered_emb.shape == (3, 4)
    assert stats["matched_nodes"] == 1
    assert stats["kept_nodes"] == 3


def test_validate_knn_inputs_rejects_too_large_k():
    with pytest.raises(ValueError, match="requires at least 31 nodes"):
        MODULE.validate_knn_inputs(30, 30)
