import importlib.util
from pathlib import Path

import polars as pl


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_embedding_knn_edges.py"
SPEC = importlib.util.spec_from_file_location("build_embedding_knn_edges", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_filter_text_rows_applies_min_lengths_and_abstract_requirement():
    df = pl.DataFrame({
        "work_id": ["W1", "W2", "W3"],
        "title": ["Good title", "Tiny", "Another title"],
        "abstract": [
            "This abstract is comfortably long.",
            "short",
            "",
        ],
    })

    filtered, stats = MODULE.filter_text_rows(
        df,
        min_text_len=20,
        min_title_len=3,
        min_abstract_len=10,
        require_abstract=True,
    )

    assert filtered["work_id"].to_list() == ["W1"]
    assert stats["input_rows"] == 3
    assert stats["kept_rows"] == 1
    assert stats["dropped_rows"] == 2
