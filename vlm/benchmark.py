"""
Performance benchmarking for VLM inference
"""

import time
import statistics
from typing import List, Dict, Optional
from pathlib import Path
import json


class VLMBenchmark:
    """
    Benchmarks VLM inference performance.
    """
    
    def __init__(self, vlm):
        """
        Initialize benchmark.
        
        Args:
            vlm: VLM model instance
        """
        self.vlm = vlm
        self.results = []
    
    def benchmark(self, image_path: str, query: str, num_runs: int = 10) -> Dict:
        """
        Benchmark single query.
        
        Args:
            image_path: Path to image
            query: Query string
            num_runs: Number of runs
        
        Returns:
            Benchmark results dictionary
        """
        timings = []
        
        for i in range(num_runs):
            start = time.time()
            result = self.vlm.generate(image_path, query)
            elapsed = time.time() - start
            timings.append(elapsed)
        
        return {
            "mean": statistics.mean(timings),
            "median": statistics.median(timings),
            "min": min(timings),
            "max": max(timings),
            "std": statistics.stdev(timings) if len(timings) > 1 else 0.0,
            "p95": sorted(timings)[int(len(timings) * 0.95)] if len(timings) > 1 else timings[0],
            "p99": sorted(timings)[int(len(timings) * 0.99)] if len(timings) > 1 else timings[0],
            "runs": num_runs
        }
    
    def benchmark_batch(self, test_cases: List[Dict], num_runs: int = 5) -> Dict:
        """
        Benchmark multiple test cases.
        
        Args:
            test_cases: List of {"image": path, "query": str}
            num_runs: Number of runs per test case
        
        Returns:
            Aggregate benchmark results
        """
        all_timings = []
        per_case_results = []
        
        for case in test_cases:
            image_path = case["image"]
            query = case["query"]
            
            case_timings = []
            for _ in range(num_runs):
                start = time.time()
                self.vlm.generate(image_path, query)
                elapsed = time.time() - start
                case_timings.append(elapsed)
                all_timings.append(elapsed)
            
            case_result = {
                "image": image_path,
                "query": query,
                "mean": statistics.mean(case_timings),
                "p95": sorted(case_timings)[int(len(case_timings) * 0.95)] if len(case_timings) > 1 else case_timings[0]
            }
            per_case_results.append(case_result)
        
        return {
            "overall": {
                "mean": statistics.mean(all_timings),
                "median": statistics.median(all_timings),
                "min": min(all_timings),
                "max": max(all_timings),
                "p95": sorted(all_timings)[int(len(all_timings) * 0.95)] if len(all_timings) > 1 else all_timings[0],
                "p99": sorted(all_timings)[int(len(all_timings) * 0.99)] if len(all_timings) > 1 else all_timings[0],
                "total_runs": len(all_timings)
            },
            "per_case": per_case_results
        }
    
    def profile_components(self, image_path: str, query: str, num_runs: int = 10) -> Dict:
        """
        Profile individual components of the pipeline.
        
        Args:
            image_path: Path to image
            query: Query string
            num_runs: Number of runs
        
        Returns:
            Component timing breakdown
        """
        from vlm.inference_optimizer import InferenceProfiler
        
        profiler = InferenceProfiler()
        
        for _ in range(num_runs):
            # Profile detection
            with profiler.profile("detection"):
                detection_results = self.vlm.detector.predict(image_path)
            
            # Profile tokenization
            with profiler.profile("tokenization"):
                prompt = self.vlm.tokenizer.encode(detection_results.get("defects", []))
            
            # Profile LLM
            with profiler.profile("llm_inference"):
                full_prompt = self.vlm._create_prompt(prompt, query)
                llm_response = self.vlm.llm.generate(full_prompt)
            
            # Profile parsing
            with profiler.profile("parsing"):
                self.vlm.parser.parse(
                    llm_response,
                    query=query,
                    detections=detection_results.get("defects", [])
                )
        
        return profiler.get_stats()
    
    def save_results(self, results: Dict, output_path: str):
        """
        Save benchmark results to file.
        
        Args:
            results: Benchmark results
            output_path: Path to save JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Benchmark results saved to {output_path}")
    
    def print_report(self, results: Dict):
        """
        Print benchmark report.
        
        Args:
            results: Benchmark results
        """
        print("\n" + "="*60)
        print("VLM Performance Benchmark")
        print("="*60)
        
        if "overall" in results:
            overall = results["overall"]
            print(f"\nOverall Performance ({overall['total_runs']} runs):")
            print(f"  Mean: {overall['mean']:.3f}s")
            print(f"  Median: {overall['median']:.3f}s")
            print(f"  Min: {overall['min']:.3f}s")
            print(f"  Max: {overall['max']:.3f}s")
            print(f"  P95: {overall['p95']:.3f}s")
            print(f"  P99: {overall['p99']:.3f}s")
            
            # Check if meets requirement
            if overall['p95'] < 2.0:
                print(f"\n✅ Meets <2s requirement (P95: {overall['p95']:.3f}s)")
            else:
                print(f"\n❌ Does NOT meet <2s requirement (P95: {overall['p95']:.3f}s)")
        
        if "per_case" in results:
            print(f"\nPer-Case Results:")
            for i, case in enumerate(results["per_case"][:5], 1):  # Show first 5
                print(f"  Case {i}: {case['mean']:.3f}s (P95: {case['p95']:.3f}s)")
        
        print("="*60)
