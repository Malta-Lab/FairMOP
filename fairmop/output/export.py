"""
Result export utilities for FairMOP.

Provides functions to serialize and deserialize FairMOP evaluation
results in JSON and CSV formats, with support for organized
metric tables and annotated data export.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


def export_results_json(
    results: Dict[str, Any],
    output_path: str,
    indent: int = 2,
    include_metadata: bool = True,
) -> str:
    """Export evaluation results to a JSON file.

    Parameters:
        results: The FairMOP results dictionary.
        output_path: Destination file path.
        indent: JSON indentation level.
        include_metadata: If True, add timestamp and version info.

    Returns:
        The absolute path of the saved file.
    """
    if include_metadata:
        from fairmop import __version__

        results = {
            "_metadata": {
                "framework": "FairMOP",
                "version": __version__,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
            **results,
        }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Handle inf/nan for JSON serialization
    def sanitize(obj):
        if isinstance(obj, float):
            if obj == float("inf"):
                return "Infinity"
            if obj == float("-inf"):
                return "-Infinity"
            if obj != obj:  # NaN check
                return "NaN"
        return obj

    class SafeEncoder(json.JSONEncoder):
        def default(self, o):
            return sanitize(o)

        def encode(self, o):
            return super().encode(self._walk(o))

        def _walk(self, o):
            if isinstance(o, dict):
                return {k: self._walk(v) for k, v in o.items()}
            if isinstance(o, list):
                return [self._walk(v) for v in o]
            return sanitize(o)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=indent, ensure_ascii=False, cls=SafeEncoder)

    abs_path = os.path.abspath(output_path)
    size_mb = os.path.getsize(abs_path) / (1024 * 1024)
    print(f"[FairMOP] Results saved to {abs_path} ({size_mb:.2f} MB)")

    return abs_path


def export_results_csv(
    results: Dict[str, Any],
    output_path: str,
    include_demographics: bool = False,
) -> str:
    """Export evaluation results as a CSV summary table.

    Each row represents one hyperparameter configuration with its
    aggregate metric values.

    Parameters:
        results: The FairMOP results dictionary.
        output_path: Destination file path.
        include_demographics: If True, include per-image demographics columns.

    Returns:
        The absolute path of the saved file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    configurations = results.get("configurations", {})
    if not configurations:
        print("[FairMOP] No configurations to export.")
        return ""

    # Collect all possible aggregate keys
    all_keys = set()
    for config_data in configurations.values():
        aggregates = config_data.get("aggregates", {})
        all_keys.update(aggregates.keys())

    # Sort keys for consistent column order
    sorted_keys = sorted(all_keys)

    # Write CSV
    fieldnames = ["configuration", "num_images"] + sorted_keys
    if include_demographics:
        fieldnames.extend(["demographics_raw"])

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for config_name, config_data in sorted(configurations.items()):
            row = {
                "configuration": config_name,
                "num_images": config_data.get("images", 0),
            }

            aggregates = config_data.get("aggregates", {})
            for key in sorted_keys:
                val = aggregates.get(key, "")
                if isinstance(val, float):
                    if val == float("inf"):
                        row[key] = "inf"
                    elif val != val:  # NaN
                        row[key] = "NaN"
                    else:
                        row[key] = f"{val:.6f}"
                else:
                    row[key] = val

            if include_demographics:
                demo = config_data.get("demographics", [])
                row["demographics_raw"] = json.dumps(demo)

            writer.writerow(row)

    abs_path = os.path.abspath(output_path)
    n_rows = len(configurations)
    print(f"[FairMOP] CSV exported to {abs_path} ({n_rows} configurations)")

    return abs_path


def load_results_json(path: str) -> Dict[str, Any]:
    """Load a FairMOP results JSON file.

    Parameters:
        path: Path to the JSON file.

    Returns:
        The results dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file has invalid structure.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Restore inf/nan values
    data = _restore_special_values(data)

    # Validate structure
    required = {"configurations"}
    if not required.issubset(data.keys()):
        # Check if it has metadata wrapper
        if "_metadata" in data and "configurations" in data:
            pass
        else:
            raise ValueError(
                f"Invalid FairMOP results file. Expected keys: {required}. "
                f"Found: {set(data.keys())}"
            )

    return data


def _restore_special_values(obj):
    """Recursively restore 'Infinity' and 'NaN' strings to float values."""
    if isinstance(obj, dict):
        return {k: _restore_special_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_restore_special_values(v) for v in obj]
    if isinstance(obj, str):
        if obj == "Infinity":
            return float("inf")
        if obj == "-Infinity":
            return float("-inf")
        if obj == "NaN":
            return float("nan")
    return obj


def export_pareto_csv(
    pareto_result,
    output_path: str,
) -> str:
    """Export Pareto analysis results to CSV.

    Parameters:
        pareto_result: A :class:`ParetoResult` instance.
        output_path: Destination file path.

    Returns:
        The absolute path of the saved file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fieldnames = [
        "configuration",
        "utility",
        "fairness",
        "is_pareto",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for point in pareto_result.all_points:
            writer.writerow(
                {
                    "configuration": point.config_name,
                    "utility": f"{point.utility:.6f}",
                    "fairness": f"{point.fairness:.6f}",
                    "is_pareto": point.is_pareto,
                }
            )

    abs_path = os.path.abspath(output_path)
    print(f"[FairMOP] Pareto CSV exported to {abs_path}")
    return abs_path
