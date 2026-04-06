"""Link-type edge construction from citation data.

Build DC (Direct Citation), BC (Bibliographic Coupling), and CC (Co-Citation)
edge tables from raw citation data, ready for Leiden clustering.

Pipeline
--------
::

    build_dc/bc/cc → filter → normalize → combine → build_graph()

Example
-------
>>> from sciscape.linkage import (
...     build_bc, build_cc, build_dc, combine_edges,
...     filter_top_k, filter_min_weight, normalize_weights,
...     LinkageConfig, CombineMethod, WeightNorm,
... )
>>>
>>> # 1. Build from citations
>>> bc = build_bc(citations, node_ids)["bc_cosine"]
>>> cc = build_cc(citations, node_ids)["cc_cosine"]
>>>
>>> # 2. Filter
>>> bc = filter_top_k(bc, k=30)
>>> cc = filter_top_k(cc, k=30)
>>>
>>> # 3. Normalize
>>> bc = normalize_weights(bc, WeightNorm.MAX)
>>> cc = normalize_weights(cc, WeightNorm.MAX)
>>>
>>> # 4. Combine
>>> edges = combine_edges(
...     {"bc": bc, "cc": cc},
...     method=CombineMethod.WEIGHTED_SUM,
...     weights={"bc": 0.7, "cc": 0.3},
... )
>>>
>>> # 5. Cluster
>>> from sciscape.clustering.graph import build_graph
>>> graph = build_graph(edges)
"""

from .builders import build_bc, build_cc, build_dc
from .combination import combine_edges, priority_fill_edges
from .config import CombineMethod, DCNormalization, LinkageConfig, Normalization
from .filters import (
    WeightNorm,
    filter_giant_component,
    filter_min_weight,
    filter_top_k,
    normalize_weights,
)
from .diagnostics import (
    complementarity_analysis,
    degree_comparison,
    edge_overlap,
    edge_stats,
    overlap_matrix,
    weight_correlation,
)

__all__ = [
    # Builders
    "build_bc",
    "build_cc",
    "build_dc",
    # Filters & normalization
    "filter_giant_component",
    "filter_min_weight",
    "filter_top_k",
    "normalize_weights",
    # Combination
    "combine_edges",
    "priority_fill_edges",
    # Diagnostics
    "complementarity_analysis",
    "degree_comparison",
    "edge_overlap",
    "edge_stats",
    "overlap_matrix",
    "weight_correlation",
    # Config / enums
    "CombineMethod",
    "DCNormalization",
    "LinkageConfig",
    "Normalization",
    "WeightNorm",
]
