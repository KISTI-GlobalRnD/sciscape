"""Tests for sciscape.clustering.partitioning — partition_class helper."""

from __future__ import annotations

import leidenalg as la
import pytest

from sciscape.clustering.partitioning import partition_class


class TestPartitionClass:
    def test_cpm_returns_cpm_partition(self):
        cls = partition_class("cpm")
        assert cls is la.CPMVertexPartition

    def test_modularity_returns_rb_partition(self):
        cls = partition_class("modularity")
        assert cls is la.RBConfigurationVertexPartition

    def test_invalid_objective_raises(self):
        with pytest.raises(ValueError, match="modularity.*cpm"):
            partition_class("invalid_objective")

    def test_case_sensitive(self):
        """Objective matching is case-sensitive (uppercase should fail)."""
        with pytest.raises(ValueError):
            partition_class("CPM")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            partition_class("")
