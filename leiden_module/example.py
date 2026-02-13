"""Example usage of the `leiden_module` pipeline."""

from __future__ import annotations

from pathlib import Path

from . import LeidenConfig, get_cluster_hierarchy, run_pipeline


def main() -> None:
    config = LeidenConfig(
        level_constraints=[
            (5, 100),
            (80, 500),
            (400, 5000),
        ],
        resolution_bounds=(1e-3, 5.0),
        progress=lambda msg: print(f"[progress] {msg}"),
        log_history=True,
    )

    tables = run_pipeline(
        zip_path=Path("../Data/KRISS_pair_links/dc_bc_cc_total_pair.zip"),
        inner_name="dc_bc_cc_total_pair.txt",
        config=config,
    )

    membership = tables.membership
    description = tables.description

    print("Cluster counts:")
    for level in tables.levels:
        column = f"cluster_{level}"
        print(f"  {level}: {membership[column].n_unique()} unique labels")
    if tables.resolutions:
        print("Resolved resolutions:")
        for level, gamma in tables.resolutions.items():
            print(f"  {level}: {gamma:.6f}")
    if tables.qualities:
        print("Modularity quality per level:")
        for level, quality in tables.qualities.items():
            print(f"  {level}: {quality:.6f}")

    membership_out = Path("./example_membership.parquet")
    membership.write_parquet(membership_out)
    print(f"Saved membership table to {membership_out}")

    description_out = Path("./example_description.parquet")
    description.write_parquet(description_out)
    print(f"Saved description table to {description_out}")

    hierarchy = get_cluster_hierarchy(tables.raw_membership, levels=tables.levels)
    level_names = list(tables.levels)
    top_label = level_names[0]
    child_label = level_names[1] if len(level_names) > 1 else "child"
    grandchild_label = level_names[2] if len(level_names) > 2 else "grandchild"

    top_level_clusters = hierarchy["clusters"]
    print("Top-level cluster overview (first 5):")
    for cluster_id, info in list(top_level_clusters.items())[:5]:
        children = info.get("children", {})
        print(
            f"  {top_label} {cluster_id}: {info['size']} nodes -> "
            f"{len(children)} {child_label} clusters"
        )
        if children:
            first_child_id, first_child_info = next(iter(children.items()))
            grandchildren = first_child_info.get("children", {})
            print(
                f"    {child_label} {first_child_id}: {first_child_info['size']} nodes -> "
                f"{len(grandchildren)} {grandchild_label} clusters"
            )


if __name__ == "__main__":
    main()
