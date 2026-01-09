"""
Inference optimization utilities for <2s inference requirement
"""

import time
from typing import Dict, List, Optional, Callable
from functools import wraps
import threading
from queue import Queue


def time_inference(func: Callable) -> Callable:
    """
    Decorator to time inference operations.
    
    Usage:
        @time_inference
        def generate_response(...):
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        if elapsed > 0.5:  # Log slow operations
            print(f"{func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper


class InferenceProfiler:
    """
    Profiles inference pipeline to identify bottlenecks.
    """
    
    def __init__(self):
        """Initialize profiler."""
        self.timings = {}
        self.call_count = {}
    
    def profile(self, component: str):
        """
        Context manager for profiling components.
        
        Usage:
            with profiler.profile("detection"):
                result = detector.predict(image)
        """
        return self._ProfileContext(self, component)
    
    class _ProfileContext:
        def __init__(self, profiler, component):
            self.profiler = profiler
            self.component = component
            self.start = None
        
        def __enter__(self):
            self.start = time.time()
            return self
        
        def __exit__(self, *args):
            elapsed = time.time() - self.start
            if self.component not in self.profiler.timings:
                self.profiler.timings[self.component] = []
                self.profiler.call_count[self.component] = 0
            self.profiler.timings[self.component].append(elapsed)
            self.profiler.call_count[self.component] += 1
    
    def get_stats(self) -> Dict[str, Dict]:
        """
        Get profiling statistics.
        
        Returns:
            Dictionary with stats for each component
        """
        stats = {}
        for component, timings in self.timings.items():
            if timings:
                stats[component] = {
                    "count": self.call_count[component],
                    "total": sum(timings),
                    "mean": sum(timings) / len(timings),
                    "min": min(timings),
                    "max": max(timings),
                    "p95": sorted(timings)[int(len(timings) * 0.95)] if len(timings) > 1 else timings[0]
                }
        return stats
    
    def print_report(self):
        """Print profiling report."""
        stats = self.get_stats()
        print("\n" + "="*60)
        print("Inference Profiling Report")
        print("="*60)
        for component, stat in stats.items():
            print(f"\n{component}:")
            print(f"  Calls: {stat['count']}")
            print(f"  Mean: {stat['mean']:.3f}s")
            print(f"  Min: {stat['min']:.3f}s")
            print(f"  Max: {stat['max']:.3f}s")
            print(f"  P95: {stat['p95']:.3f}s")
        print("="*60)


class AsyncInference:
    """
    Asynchronous inference for non-blocking operations.
    """
    
    def __init__(self, vlm, max_workers: int = 2):
        """
        Initialize async inference.
        
        Args:
            vlm: VLM model instance
            max_workers: Maximum number of worker threads
        """
        self.vlm = vlm
        self.max_workers = max_workers
        self.queue = Queue()
        self.results = {}
        self.threads = []
    
    def submit(self, image_path: str, query: str, request_id: str):
        """
        Submit inference request.
        
        Args:
            image_path: Path to image
            query: Query string
            request_id: Unique request identifier
        """
        self.queue.put((request_id, image_path, query))
    
    def get_result(self, request_id: str, timeout: float = 5.0) -> Optional[Dict]:
        """
        Get result for request.
        
        Args:
            request_id: Request identifier
            timeout: Timeout in seconds
        
        Returns:
            Result dictionary or None if timeout
        """
        start = time.time()
        while time.time() - start < timeout:
            if request_id in self.results:
                result = self.results.pop(request_id)
                return result
            time.sleep(0.1)
        return None
    
    def _worker(self):
        """Worker thread for processing requests."""
        while True:
            try:
                request_id, image_path, query = self.queue.get(timeout=1.0)
                result = self.vlm.generate(image_path, query)
                self.results[request_id] = result
                self.queue.task_done()
            except:
                break


def optimize_for_speed(config: Dict) -> Dict:
    """
    Apply speed optimizations to configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Optimized configuration
    """
    optimized = config.copy()
    
    # Reduce LLM temperature for faster generation
    if "llm_temperature" in optimized:
        optimized["llm_temperature"] = min(optimized["llm_temperature"], 0.1)
    
    # Reduce max tokens
    if "llm_max_tokens" in optimized:
        optimized["llm_max_tokens"] = min(optimized["llm_max_tokens"], 256)
    
    # Enable quantization
    if "use_quantization" not in optimized:
        optimized["use_quantization"] = True
    
    # Enable ONNX
    if "use_onnx" not in optimized:
        optimized["use_onnx"] = True
    
    return optimized
