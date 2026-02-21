"""
FairMOP Example – Evaluate Existing Results.

Shows how to load and analyze pre-existing FairMOP results JSON files,
build Pareto frontiers, and compare multiple models.

This is useful when you already have generated images and evaluation
data (e.g., from the look_here/examples/ directory).

Usage:
    python examples/analyze_results.py
"""

import os
import sys

from fairmop.output.export import load_results_json
from fairmop.output.pareto import build_pareto_from_results, compute_hypervolume


def main():
    # Path to example results
    examples_dir = os.path.join(
        os.path.dirname(__file__), "..", "look_here", "examples"
    )

    results_files = {
        "Fair Diffusion": os.path.join(examples_dir, "nurse_fairdiffusion_5000.json"),
        "FLUX (5000)": os.path.join(examples_dir, "nurse_flux_5000_completo.json"),
        "FLUX (Default)": os.path.join(
            examples_dir, "nurse_flux_default_completo.json"
        ),
    }

    print("=" * 60)
    print("  FairMOP – Pareto Analysis of Existing Results")
    print("=" * 60)

    for model_name, filepath in results_files.items():
        if not os.path.exists(filepath):
            print(f"\n[SKIP] {model_name}: file not found at {filepath}")
            continue

        print(f"\n--- {model_name} ---")

        results = load_results_json(filepath)

        n_configs = len(results.get("configurations", {}))
        print(f"  Configurations: {n_configs}")
        print(f"  Topic: {results.get('topic', '?')}")

        # Check available metrics
        if n_configs > 0:
            first = list(results["configurations"].values())[0]
            agg_keys = list(first.get("aggregates", {}).keys())
            print(f"  Available metrics: {agg_keys}")

            # Try to build Pareto
            utility_key = None
            fairness_key = None

            for k in ["avg_clip_score", "fid_score", "prdc_score"]:
                if k in agg_keys:
                    utility_key = k
                    break

            for k in ["gender_entropy", "entropy_fairness", "primary_entropy"]:
                if k in agg_keys:
                    fairness_key = k
                    break

            if utility_key and fairness_key:
                pareto = build_pareto_from_results(
                    results,
                    utility_metric=utility_key,
                    fairness_metric=fairness_key,
                    utility_invert=(utility_key == "fid_score"),
                )
                print(f"  Pareto-optimal: {pareto.n_pareto}/{pareto.n_configurations}")

                # Hypervolume
                hv = compute_hypervolume(pareto.pareto_points)
                print(f"  Hypervolume indicator: {hv:.6f}")

                if pareto.pareto_points:
                    print("  Pareto configurations:")
                    for p in pareto.pareto_points[:5]:
                        print(
                            f"    {p.config_name}: "
                            f"utility={p.utility:.4f}, fairness={p.fairness:.4f}"
                        )
                    if pareto.n_pareto > 5:
                        print(f"    ... and {pareto.n_pareto - 5} more")
            else:
                print(
                    f"  Could not find utility ({utility_key}) or "
                    f"fairness ({fairness_key}) metric for Pareto analysis."
                )

    print(f"\n{'=' * 60}")
    print("  To visualize interactively, run:")
    print("  python -m fairmop dashboard")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
