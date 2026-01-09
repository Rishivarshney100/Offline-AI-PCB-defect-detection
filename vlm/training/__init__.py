"""
Training module for Detection-Grounded VLM
"""

from vlm.training.qa_generator import QAGenerator
from vlm.training.trainer import VLMTrainer
from vlm.training.augmentation import QAAugmentation

__all__ = ['QAGenerator', 'VLMTrainer', 'QAAugmentation']
