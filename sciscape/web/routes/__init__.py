"""Web API route modules.

Future split target — routes will be separated from app.py into:
  - jobs.py: query, status, stream, list, download
  - analysis.py: network, temporal, bridge, term-net, treemap, consensus, quality
  - labels.py: LLM labels, merges, abbreviations
  - export.py: GEXF/GraphML, what-if

Currently all routes remain in app.py. This package is prepared for
incremental migration.
"""
