"""
Metrics dashboard for visualizing VLM performance
"""

from typing import Dict, List
import json
from pathlib import Path


class MetricsDashboard:
    """
    Creates visualizations and reports for VLM metrics.
    """
    
    def __init__(self, metrics_file: Optional[str] = None):
        """
        Initialize dashboard.
        
        Args:
            metrics_file: Path to saved metrics JSON
        """
        self.metrics_file = metrics_file
        self.metrics = None
        
        if metrics_file and Path(metrics_file).exists():
            self.load_metrics(metrics_file)
    
    def load_metrics(self, metrics_file: str):
        """
        Load metrics from file.
        
        Args:
            metrics_file: Path to metrics JSON
        """
        with open(metrics_file, 'r') as f:
            self.metrics = json.load(f)
    
    def create_summary_report(self, output_path: str):
        """
        Create summary report.
        
        Args:
            output_path: Path to save report
        """
        if not self.metrics:
            print("No metrics loaded")
            return
        
        report = f"""
# VLM Performance Summary Report

## Overall Metrics

### Counting Accuracy
- Accuracy: {self.metrics.get('counting_accuracy', {}).get('accuracy', 0):.2%}
- Correct: {self.metrics.get('counting_accuracy', {}).get('correct', 0)}
- Total: {self.metrics.get('counting_accuracy', {}).get('total', 0)}

### Localization Precision
- Mean IoU: {self.metrics.get('localization_precision', {}).get('mean_iou', 0):.3f}
- Mean Center Error: {self.metrics.get('localization_precision', {}).get('mean_center_error', 0):.2f}px
- P95 Center Error: {self.metrics.get('localization_precision', {}).get('p95_center_error', 0):.2f}px

### Hallucination Rate
- Rate: {self.metrics.get('hallucination_rate', {}).get('rate', 0):.2%}
- Count: {self.metrics.get('hallucination_rate', {}).get('count', 0)}
- Total: {self.metrics.get('hallucination_rate', {}).get('total', 0)}

### Inference Latency
- Mean: {self.metrics.get('latency', {}).get('mean', 0):.3f}s
- P95: {self.metrics.get('latency', {}).get('p95', 0):.3f}s
- P99: {self.metrics.get('latency', {}).get('p99', 0):.3f}s

## Requirements Check

- [ ] Counting Accuracy >95%: {'✅' if self.metrics.get('counting_accuracy', {}).get('accuracy', 0) >= 0.95 else '❌'}
- [ ] Localization Error <5px: {'✅' if self.metrics.get('localization_precision', {}).get('mean_center_error', 999) < 5.0 else '❌'}
- [ ] Hallucination Rate <1%: {'✅' if self.metrics.get('hallucination_rate', {}).get('rate', 1) < 0.01 else '❌'}
- [ ] Inference <2s (P95): {'✅' if self.metrics.get('latency', {}).get('p95', 999) < 2.0 else '❌'}
"""
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"Summary report saved to {output_path}")
    
    def visualize_latency_distribution(self, output_path: str):
        """
        Create latency distribution visualization.
        
        Args:
            output_path: Path to save visualization
        """
        try:
            import matplotlib.pyplot as plt
            
            if not self.metrics or "latency" not in self.metrics:
                print("Latency data not available")
                return
            
            # This would create a histogram
            # Placeholder for now
            print(f"Latency visualization would be saved to {output_path}")
        
        except ImportError:
            print("matplotlib not available for visualization")
