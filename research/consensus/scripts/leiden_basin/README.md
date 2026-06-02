# Leiden Basin Scripts

Target location for Leiden basin, hysteresis, route, wall, operator, and
evidence-panel scripts.

Second-level buckets:

| Bucket | Intended content |
|---|---|
| `basin_signatures/` | Basin signatures, selector contracts, mode tradeoff, and general basin diagnostics |
| `transition_routes/` | Transition, route, wall-route, pathway, tunneling, and direct-pair scripts |
| `operator_probes/` | Attachment-margin, aligned-core, handle, selector, gate, recovery, and polish probes |
| `evidence_panels/` | Reviews, audits, field eligibility, relation taxonomy, phase panels, and claim evidence |
| `materialization/` | Cache, membership, prepare, join, and materialization scripts |
| `hysteresis/` | Leiden hysteresis runs, monitors, and graph materialization |
| `demo/` | Tiny controlled Leiden + CPM demo graphs and baseline seed sweeps |

The largest buckets are split once more. `operator_probes/` uses
`selector_sources/`, `selector_signals/`, `attachment_margin/`, `aligned_core/`,
`joint_bundle/`, `gate_release/`, `post_gate_recovery/`, and `polish_elbow/`.
`transition_routes/` uses `closure_context/`, `transition_operators/`,
`transition_diagnostics/`, `route_wall/`, `route_gate_panels/`,
`tunneling_pathways/`, and `route_reviews/`. `basin_signatures/` uses
`signature_detection/`, `portfolio_contracts/`, `trajectory_failure/`,
`branch_growth/`, `local_modes/`, and `endpoint_flips/`. `evidence_panels/`
uses `audits/`, `field_eligibility/`, `relation_taxonomy/`, `phase_panels/`,
`portfolio_evidence/`, and `review_panels/`.
