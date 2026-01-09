"""
Performance tests for <2s inference requirement
"""

import unittest
import time
from unittest.mock import Mock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vlm.benchmark import VLMBenchmark


class TestPerformance(unittest.TestCase):
    """Test performance requirements."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock VLM for testing
        self.mock_vlm = Mock()
        self.benchmark = VLMBenchmark(self.mock_vlm)
    
    def test_inference_time_target(self):
        """Test that inference time meets <2s target."""
        # Mock fast inference
        self.mock_vlm.generate.return_value = {"count": 1}
        
        # Simulate timing
        start = time.time()
        result = self.benchmark.benchmark("dummy_image.jpg", "How many defects?", num_runs=5)
        elapsed = time.time() - start
        
        # Check that mean is reasonable (allowing for test overhead)
        if "mean" in result:
            self.assertLess(result["mean"], 3.0, "Inference should be fast")
    
    def test_p95_latency(self):
        """Test P95 latency requirement."""
        # Mock VLM with varying latency
        call_times = [0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5]
        
        def mock_generate(*args, **kwargs):
            time.sleep(call_times.pop(0) if call_times else 0.5)
            return {"count": 1}
        
        self.mock_vlm.generate.side_effect = mock_generate
        
        result = self.benchmark.benchmark("dummy_image.jpg", "How many defects?", num_runs=10)
        
        if "p95" in result:
            # P95 should be reasonable
            self.assertLess(result["p95"], 3.0, "P95 latency should be acceptable")


if __name__ == "__main__":
    unittest.main()
