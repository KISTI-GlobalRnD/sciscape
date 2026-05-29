from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "consensus"
    / "scripts"
    / "leiden_basin"
    / "transition_routes"
    / "transition_operators"
    / "run_leiden_basin_transition_operator_pilot.py"
)


def _load_module():
    script_dir = SCRIPT_PATH.parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location(
        "run_leiden_basin_transition_operator_pilot_for_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_changed_support_nodes_is_invariant_to_label_permutation():
    module = _load_module()
    baseline = np.asarray([10, 10, 20, 20, 30], dtype=np.uint64)
    relabeled_same = np.asarray([1, 1, 2, 2, 3], dtype=np.uint64)

    changed = module.changed_support_nodes(baseline, relabeled_same)

    assert changed.tolist() == []


def test_changed_support_nodes_detects_split_and_merge_footprint():
    module = _load_module()
    baseline = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.uint64)
    membership = np.asarray([0, 3, 1, 1, 3, 4], dtype=np.uint64)

    changed = module.changed_support_nodes(baseline, membership)

    assert changed.tolist() == [1, 4, 5]


def test_transplant_support_groups_offsets_donor_labels_without_collision():
    module = _load_module()
    base = np.asarray([0, 0, 1, 1, 2], dtype=np.uint64)
    donor = np.asarray([9, 9, 9, 8, 8], dtype=np.uint64)
    support = np.asarray([0, 2, 3], dtype=np.uint32)

    transplanted = module.transplant_support_groups(base, donor, support)

    assert transplanted[1] == 0
    assert transplanted[4] == 2
    assert transplanted[0] == transplanted[2]
    assert transplanted[3] != transplanted[0]
    assert int(transplanted[0]) > int(base.max())


def test_fixed_outside_marks_only_support_nodes_mutable():
    module = _load_module()

    fixed = module.fixed_outside(5, np.asarray([1, 3], dtype=np.uint32))

    assert fixed.tolist() == [True, False, True, False, True]
