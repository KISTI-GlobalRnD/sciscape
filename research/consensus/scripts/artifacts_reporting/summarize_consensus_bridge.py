"""Summarize citation-consensus vs all-consensus bridge metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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


from _common import save_json, select_best_single_result

def _load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def _index_results(results: list[dict]) -> dict[str, dict]:
    return {row["method"]: row for row in results if row.get("method")}

def _protocol_label(payload: dict) -> str | None:
    protocol = payload.get("protocol")
    if protocol:
        return protocol
    budget_mode = payload.get("budget_mode")
    if budget_mode == "effective_k":
        return "candidate_budget_matched"
    if budget_mode == "top_k":
        return "practical_top_k"
    return None

def _is_current_payload(payload: dict) -> bool:
    return bool(payload.get("protocol")) and bool(payload.get("layer_paths"))

def main() -> None:
    parser = argparse.ArgumentParser(description="Bridge summary for citation_consensus vs all_consensus")
    parser.add_argument("comparison_json", nargs="+", type=Path, help="One or more *_comparison.json files")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--allow-legacy", action="store_true", help="Allow payloads without explicit protocol/layer_paths")
    args = parser.parse_args()

    rows: list[dict] = []
    for path in args.comparison_json:
        payload = _load_payload(path)
        if not _is_current_payload(payload):
            if not args.allow_legacy:
                raise RuntimeError(f"Legacy or incomplete comparison payload requires --allow-legacy: {path}")
        results = payload.get("results", [])
        by_method = _index_results(results)
        best_single = select_best_single_result(results)
        citation = by_method.get("citation_consensus")
        all_cons = by_method.get("all_consensus")
        rows.append(
            {
                "source_json": str(path),
                "field": payload.get("field"),
                "protocol": _protocol_label(payload),
                "budget_mode": payload.get("budget_mode"),
                "effective_k": payload.get("effective_k"),
                "top_k": payload.get("top_k"),
                "emb_mode": payload.get("emb_mode"),
                "emb_path": payload.get("emb_path"),
                "best_single_method": best_single["method"] if best_single else None,
                "best_single_ami": best_single["ami_mean"] if best_single else None,
                "citation_consensus_ami": citation["ami_mean"] if citation else None,
                "all_consensus_ami": all_cons["ami_mean"] if all_cons else None,
                "citation_vs_best_single": (
                    citation["ami_mean"] - best_single["ami_mean"]
                    if citation and best_single else None
                ),
                "all_vs_best_single": (
                    all_cons["ami_mean"] - best_single["ami_mean"]
                    if all_cons and best_single else None
                ),
                "citation_vs_all": (
                    citation["ami_mean"] - all_cons["ami_mean"]
                    if citation and all_cons else None
                ),
                "citation_edges": citation["n_edges"] if citation else None,
                "all_edges": all_cons["n_edges"] if all_cons else None,
                "citation_clusters": citation["n_clusters"] if citation else None,
                "all_clusters": all_cons["n_clusters"] if all_cons else None,
            }
        )

    save_json({"rows": rows}, args.output)

if __name__ == "__main__":
    main()
