"""
Application package for running PINN Heat workflows:
- ETL (data loading & sampling)
- Training (ADAM + LBFGS)
- Inference (grid predictions)
- Evaluation (metrics)
- Visualization (plots)
"""
__all__ = ["cli"]
