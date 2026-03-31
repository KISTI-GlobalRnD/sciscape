#!/usr/bin/env python3
"""KRISS-specific wrapper for the SciScape landscape pipeline.

Usage:
    python scripts/landscape_kriss.py [--n-nodes 100000] [--seed 42]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sciscape.landscape import LandscapeConfig, run_landscape

# ---------------------------------------------------------------------------
# KRISS-specific paths
# ---------------------------------------------------------------------------
KRISS_DIR = Path.home() / "Desktop/Workspace/1.4.2.KRISS"
EDGE_PATH = KRISS_DIR / "Data" / "KRISS_pair_links" / "dc_bc_cc_total_pair.txt"
ABSTRACT_PATH = KRISS_DIR / "Data" / "FINAL_title_abstract_pubyear.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "landscape"


def main():
    parser = argparse.ArgumentParser(description="KRISS landscape pipeline")
    parser.add_argument("--n-nodes", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Ignore cached results")
    args = parser.parse_args()

    # Set up logging
    log_file = OUTPUT_DIR / "run.log"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        ],
    )

    assert EDGE_PATH.exists(), f"Edge list not found: {EDGE_PATH}"
    assert ABSTRACT_PATH.exists(), f"Abstract file not found: {ABSTRACT_PATH}"

    cfg = LandscapeConfig(
        n_target_nodes=args.n_nodes,
        seed=args.seed,
        force=args.force,
        report_title="KRISS Landscape",
    )

    run_landscape(EDGE_PATH, ABSTRACT_PATH, OUTPUT_DIR, config=cfg)


if __name__ == "__main__":
    main()
