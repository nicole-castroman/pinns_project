"""
Inference utilities (regular grid sampling for visualization and export).
"""
from .grids import predict_on_grid
# Import run_inference from the sibling infer_runner.py module
from ..infer_runner import run_inference

__all__ = ["predict_on_grid", "run_inference"]
