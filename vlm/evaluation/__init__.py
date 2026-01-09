"""
Evaluation module for VLM
"""

from vlm.evaluation.evaluator import VLMEvaluator
from vlm.evaluation.validate import validate_on_dataset, validate_edge_cases
from vlm.evaluation.metrics_dashboard import MetricsDashboard

__all__ = ['VLMEvaluator', 'validate_on_dataset', 'validate_edge_cases', 'MetricsDashboard']
