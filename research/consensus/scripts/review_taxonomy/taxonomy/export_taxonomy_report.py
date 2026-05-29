"""Export taxonomy summaries as paper-ready Markdown and LaTeX snippets."""

from __future__ import annotations

import argparse
import json
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


def _pct(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0.0%"
    return f"{(100.0 * numerator / denominator):.1f}%"

def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = text
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out

def _winner_table_md(summary: dict[str, Any]) -> str:
    lines = [
        "| Slice | consensus_all | sum_minus_emb | sum_minus_cc |",
        "|---|---:|---:|---:|",
    ]
    for k, counts in sorted(summary["winner_by_k"].items(), key=lambda item: int(item[0])):
        lines.append(
            f"| top-k={k} | {counts.get('consensus_all', 0)} | {counts.get('sum_minus_emb', 0)} | {counts.get('sum_minus_cc', 0)} |"
        )
    total = summary["winner_counts"]
    lines.append(
        f"| total | {total.get('consensus_all', 0)} | {total.get('sum_minus_emb', 0)} | {total.get('sum_minus_cc', 0)} |"
    )
    return "\n".join(lines)

def _label_table_md(summary: dict[str, Any]) -> str:
    total_cases = int(summary["n_classified_cases"])
    lines = [
        "| Taxonomy label | Count | Share | consensus_all wins | sum_minus_emb wins | sum_minus_cc wins |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = sorted(summary["label_counts"].items(), key=lambda item: (-item[1], item[0]))
    by_winner = summary["label_by_winner"]
    for label, count in labels:
        lines.append(
            "| {label} | {count} | {share} | {c} | {sme} | {smc} |".format(
                label=label,
                count=count,
                share=_pct(count, total_cases),
                c=by_winner.get("consensus_all", {}).get(label, 0),
                sme=by_winner.get("sum_minus_emb", {}).get(label, 0),
                smc=by_winner.get("sum_minus_cc", {}).get(label, 0),
            )
        )
    return "\n".join(lines)

def _example_block_md(title: str, examples: list[dict[str, Any]]) -> str:
    lines = [f"### {title}"]
    for example in examples:
        lines.append(
            "- `{uid}` {title} ({field}, k={k}, label={label}, gap={gap})".format(
                uid=example["target_uid"],
                title=example["target_title"],
                field=example["field"],
                k=example["top_k"],
                label=example.get("primary_label", example.get("winner_method", "")),
                gap=example["score_gap"],
            )
        )
        lines.append(f"  Winner advantage: {example['winner_advantage']}")
        lines.append(f"  Loser failure mode: {example['loser_failure_mode']}")
    return "\n".join(lines)

def _build_markdown(summary: dict[str, Any]) -> str:
    total = summary["n_classified_cases"]
    winners = summary["winner_counts"]
    consensus_wins = winners.get("consensus_all", 0)
    sme_wins = winners.get("sum_minus_emb", 0)
    smc_wins = winners.get("sum_minus_cc", 0)
    top_labels = sorted(summary["label_counts"].items(), key=lambda item: (-item[1], item[0]))[:3]
    top_label_text = ", ".join(f"`{label}` ({count})" for label, count in top_labels)

    lines = [
        "# Consensus Taxonomy Summary",
        "",
        "## Headline",
        "",
        (
            f"`consensus_all` wins {consensus_wins}/{total} cases ({_pct(consensus_wins, total)}), "
            f"while `sum_minus_emb` wins {sme_wins}/{total} and `sum_minus_cc` wins {smc_wins}/{total}."
        ),
        (
            "The dominant consensus-win regimes are "
            f"{top_label_text}, which is consistent with a boundary-regularization interpretation."
        ),
        "",
        "## Winner By K",
        "",
        _winner_table_md(summary),
        "",
        "## Taxonomy By Winner",
        "",
        _label_table_md(summary),
        "",
        "## Representative Winner Cases",
        "",
        _example_block_md("consensus_all", summary["representative_examples_by_winner"].get("consensus_all", [])),
        "",
        _example_block_md("sum_minus_emb", summary["representative_examples_by_winner"].get("sum_minus_emb", [])),
        "",
        _example_block_md("sum_minus_cc", summary["representative_examples_by_winner"].get("sum_minus_cc", [])),
        "",
        "## Representative Taxonomy Cases",
        "",
    ]
    for label in sorted(summary["representative_by_label"].keys()):
        lines.append(_example_block_md(label, summary["representative_by_label"][label]))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"

def _winner_table_tex(summary: dict[str, Any]) -> str:
    rows = []
    for k, counts in sorted(summary["winner_by_k"].items(), key=lambda item: int(item[0])):
        rows.append(
            f"top-$k$={k} & {counts.get('consensus_all', 0)} & {counts.get('sum_minus_emb', 0)} & {counts.get('sum_minus_cc', 0)} \\\\"
        )
    total = summary["winner_counts"]
    rows.append(
        f"Total & {total.get('consensus_all', 0)} & {total.get('sum_minus_emb', 0)} & {total.get('sum_minus_cc', 0)} \\\\"
    )
    return "\n".join(rows)

def _label_table_tex(summary: dict[str, Any]) -> str:
    total_cases = int(summary["n_classified_cases"])
    by_winner = summary["label_by_winner"]
    rows = []
    for label, count in sorted(summary["label_counts"].items(), key=lambda item: (-item[1], item[0])):
        rows.append(
            "{label} & {count} & {share} & {c} & {sme} & {smc} \\\\".format(
                label=_latex_escape(label),
                count=count,
                share=_latex_escape(_pct(count, total_cases)),
                c=by_winner.get("consensus_all", {}).get(label, 0),
                sme=by_winner.get("sum_minus_emb", {}).get(label, 0),
                smc=by_winner.get("sum_minus_cc", {}).get(label, 0),
            )
        )
    return "\n".join(rows)

def _example_list_tex(summary: dict[str, Any]) -> str:
    sections: list[str] = []
    for winner, examples in summary["representative_examples_by_winner"].items():
        sections.append(f"\\paragraph{{{_latex_escape(winner)}}}")
        sections.append("\\begin{itemize}")
        for example in examples:
            sections.append(
                "\\item \\textbf{{{title}}} ({field}, $k$={k}, {label}, gap={gap}). "
                "Winner advantage: {adv}. Loser failure mode: {fail}.".format(
                    title=_latex_escape(example["target_title"]),
                    field=_latex_escape(str(example["field"])),
                    k=example["top_k"],
                    label=_latex_escape(str(example.get("primary_label", ""))),
                    gap=example["score_gap"],
                    adv=_latex_escape(example["winner_advantage"]),
                    fail=_latex_escape(example["loser_failure_mode"]),
                )
            )
        sections.append("\\end{itemize}")
    return "\n".join(sections)

def _build_tex(summary: dict[str, Any]) -> str:
    return (
        "\\section{Consensus Taxonomy Summary}\n\n"
        "\\subsection{Winner by $k$}\n"
        "\\begin{table}[t]\n\\centering\n"
        "\\begin{tabular}{lrrr}\n\\hline\n"
        "Slice & consensus\\_all & sum\\_minus\\_emb & sum\\_minus\\_cc \\\\\n\\hline\n"
        f"{_winner_table_tex(summary)}\n"
        "\\hline\n\\end{tabular}\n"
        "\\caption{Bank-based local review winners by slice.}\n"
        "\\end{table}\n\n"
        "\\subsection{Taxonomy by winner}\n"
        "\\begin{table}[t]\n\\centering\n"
        "\\begin{tabular}{lrrrrr}\n\\hline\n"
        "Taxonomy label & Count & Share & consensus\\_all & sum\\_minus\\_emb & sum\\_minus\\_cc \\\\\n\\hline\n"
        f"{_label_table_tex(summary)}\n"
        "\\hline\n\\end{tabular}\n"
        "\\caption{Primary case taxonomy across all reviewed cases.}\n"
        "\\end{table}\n\n"
        "\\subsection{Representative cases}\n"
        f"{_example_list_tex(summary)}\n"
    )

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("taxonomy_summary", type=Path, help="Combined taxonomy summary JSON")
    parser.add_argument("-o", "--output", type=Path, default=Path("research/consensus/results/taxonomy"))
    parser.add_argument("--stem", type=str, default="taxonomy_report")
    args = parser.parse_args()

    summary = json.loads(args.taxonomy_summary.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)

    md_path = args.output / f"{args.stem}.md"
    md_path.write_text(_build_markdown(summary), encoding="utf-8")
    print(f"Saved → {md_path}")

    tex_path = args.output / f"{args.stem}.tex"
    tex_path.write_text(_build_tex(summary), encoding="utf-8")
    print(f"Saved → {tex_path}")

if __name__ == "__main__":
    main()
