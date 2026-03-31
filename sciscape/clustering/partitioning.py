"""Partition helpers shared between clustering modules."""

from __future__ import annotations

import leidenalg as la


def partition_class(objective: str):
    if objective == "modularity":
        return la.RBConfigurationVertexPartition
    if objective == "cpm":
        return la.CPMVertexPartition
    raise ValueError("objective must be 'modularity' or 'cpm'")


__all__ = ["partition_class"]
