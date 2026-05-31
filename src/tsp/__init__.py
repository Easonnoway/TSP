"""TSP (Tree-like Self-Play) for Secure Code LLMs.

A training framework that reframes secure code generation as a targeted
self-play process, corresponding to Algorithm 1 in the paper.
"""

from .annotate import annotate_dataset, parse_risk_nodes
from .generate import generate_branches
from .pairs import build_preference_pairs, convert_to_training_format
from .train import train_tsp

__all__ = [
    "annotate_dataset",
    "parse_risk_nodes",
    "generate_branches",
    "build_preference_pairs",
    "convert_to_training_format",
    "train_tsp",
]
