"""
Module 4 – Output.

Provides Pareto frontier identification, interactive Streamlit
visualization dashboard, and result export utilities.
"""

from fairmop.output.export import (
    export_results_csv,
    export_results_json,
    load_results_json,
)
from fairmop.output.pareto import (
    ParetoResult,
    find_pareto_frontier,
)

__all__ = [
    "find_pareto_frontier",
    "ParetoResult",
    "export_results_json",
    "export_results_csv",
    "load_results_json",
]
