"""
Evaluation framework for VLM performance metrics
"""

from typing import List, Dict, Optional
import numpy as np
from pathlib import Path


class VLMEvaluator:
    """
    Evaluates VLM performance on multiple metrics.
    """
    
    def __init__(self):
        """Initialize evaluator."""
        self.results = {
            "counting_accuracy": [],
            "localization_precision": [],
            "hallucination_events": [],
            "latency": []
        }
    
    def evaluate_counting_accuracy(self, 
                                   predicted: Dict,
                                   ground_truth: Dict) -> bool:
        """
        Evaluate counting accuracy.
        
        Args:
            predicted: Predicted response
            ground_truth: Ground truth answer
        
        Returns:
            True if count matches exactly
        """
        pred_count = predicted.get("count", 0)
        gt_count = ground_truth.get("count", 0)
        
        is_correct = (pred_count == gt_count)
        self.results["counting_accuracy"].append(is_correct)
        
        return is_correct
    
    def evaluate_localization_precision(self,
                                        predicted: Dict,
                                        ground_truth: Dict,
                                        detections: List[Dict],
                                        iou_threshold: float = 0.5) -> Dict:
        """
        Evaluate localization precision.
        
        Args:
            predicted: Predicted response
            ground_truth: Ground truth answer
            detections: Detection results
            iou_threshold: IoU threshold for matching
        
        Returns:
            Dictionary with precision metrics
        """
        pred_locations = predicted.get("locations", [])
        gt_locations = ground_truth.get("locations", [])
        
        if not pred_locations and not gt_locations:
            return {"iou": 1.0, "center_error": 0.0, "matched": True}
        
        if len(pred_locations) != len(gt_locations):
            # Count mismatch
            return {"iou": 0.0, "center_error": float('inf'), "matched": False}
        
        # Calculate IoU and center error for each location
        ious = []
        center_errors = []
        
        for pred_loc, gt_loc in zip(pred_locations, gt_locations):
            # Find matching detection
            pred_center = np.array(pred_loc)
            gt_center = np.array(gt_loc)
            
            # Center error
            center_error = np.linalg.norm(pred_center - gt_center)
            center_errors.append(center_error)
            
            # IoU calculation (simplified - would need bboxes)
            # For now, use distance-based matching
            if center_error < 10.0:  # 10 pixel tolerance
                ious.append(1.0)
            else:
                ious.append(0.0)
        
        avg_iou = np.mean(ious) if ious else 0.0
        avg_center_error = np.mean(center_errors) if center_errors else float('inf')
        
        result = {
            "iou": avg_iou,
            "center_error": avg_center_error,
            "matched": avg_iou >= iou_threshold
        }
        
        self.results["localization_precision"].append(result)
        
        return result
    
    def evaluate_hallucination(self,
                              predicted: Dict,
                              detections: List[Dict],
                              query: str) -> Dict:
        """
        Evaluate hallucination.
        
        Args:
            predicted: Predicted response
            detections: Actual detections
            query: Original query
        
        Returns:
            Hallucination detection flags
        """
        from vlm.hallucination_control import HallucinationDetector
        
        detector = HallucinationDetector()
        flags = detector.detect(predicted, detections, query)
        is_hallucination = detector.is_hallucination(flags)
        
        if is_hallucination:
            self.results["hallucination_events"].append({
                "query": query,
                "predicted": predicted,
                "flags": flags
            })
        
        return flags
    
    def evaluate_latency(self, latency: float):
        """
        Record inference latency.
        
        Args:
            latency: Inference time in seconds
        """
        self.results["latency"].append(latency)
    
    def get_metrics(self) -> Dict:
        """
        Get aggregated evaluation metrics.
        
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Counting accuracy
        if self.results["counting_accuracy"]:
            correct = sum(self.results["counting_accuracy"])
            total = len(self.results["counting_accuracy"])
            metrics["counting_accuracy"] = {
                "accuracy": correct / total,
                "correct": correct,
                "total": total
            }
        
        # Localization precision
        if self.results["localization_precision"]:
            ious = [r["iou"] for r in self.results["localization_precision"]]
            center_errors = [r["center_error"] for r in self.results["localization_precision"]]
            
            metrics["localization_precision"] = {
                "mean_iou": np.mean(ious),
                "mean_center_error": np.mean(center_errors),
                "p95_center_error": np.percentile(center_errors, 95) if center_errors else 0.0
            }
        
        # Hallucination rate
        total_queries = len(self.results["counting_accuracy"]) if self.results["counting_accuracy"] else 0
        hallucination_count = len(self.results["hallucination_events"])
        metrics["hallucination_rate"] = {
            "rate": hallucination_count / total_queries if total_queries > 0 else 0.0,
            "count": hallucination_count,
            "total": total_queries
        }
        
        # Latency
        if self.results["latency"]:
            latencies = self.results["latency"]
            metrics["latency"] = {
                "mean": np.mean(latencies),
                "median": np.median(latencies),
                "p95": np.percentile(latencies, 95),
                "p99": np.percentile(latencies, 99),
                "min": np.min(latencies),
                "max": np.max(latencies)
            }
        
        return metrics
    
    def print_report(self):
        """Print evaluation report."""
        metrics = self.get_metrics()
        
        print("\n" + "=" * 60)
        print("VLM Evaluation Report")
        print("=" * 60)
        
        if "counting_accuracy" in metrics:
            acc = metrics["counting_accuracy"]
            print(f"\nCounting Accuracy: {acc['accuracy']:.2%} ({acc['correct']}/{acc['total']})")
            if acc['accuracy'] >= 0.95:
                print("  ✅ Meets >95% requirement")
            else:
                print("  ❌ Does NOT meet >95% requirement")
        
        if "localization_precision" in metrics:
            loc = metrics["localization_precision"]
            print(f"\nLocalization Precision:")
            print(f"  Mean IoU: {loc['mean_iou']:.3f}")
            print(f"  Mean Center Error: {loc['mean_center_error']:.2f}px")
            print(f"  P95 Center Error: {loc['p95_center_error']:.2f}px")
            if loc['mean_center_error'] < 5.0:
                print("  ✅ Meets <5px requirement")
            else:
                print("  ❌ Does NOT meet <5px requirement")
        
        if "hallucination_rate" in metrics:
            hall = metrics["hallucination_rate"]
            print(f"\nHallucination Rate: {hall['rate']:.2%} ({hall['count']}/{hall['total']})")
            if hall['rate'] < 0.01:
                print("  ✅ Meets <1% requirement")
            else:
                print("  ❌ Does NOT meet <1% requirement")
        
        if "latency" in metrics:
            lat = metrics["latency"]
            print(f"\nInference Latency:")
            print(f"  Mean: {lat['mean']:.3f}s")
            print(f"  Median: {lat['median']:.3f}s")
            print(f"  P95: {lat['p95']:.3f}s")
            print(f"  P99: {lat['p99']:.3f}s")
            if lat['p95'] < 2.0:
                print("  ✅ Meets <2s requirement")
            else:
                print("  ❌ Does NOT meet <2s requirement")
        
        print("=" * 60)
    
    def reset(self):
        """Reset evaluation results."""
        self.results = {
            "counting_accuracy": [],
            "localization_precision": [],
            "hallucination_events": [],
            "latency": []
        }
