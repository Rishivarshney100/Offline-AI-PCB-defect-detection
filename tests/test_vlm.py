"""
Unit tests for VLM components
"""

import unittest
from unittest.mock import Mock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vlm.tokenizer import DefectTokenizer
from vlm.output_parser import OutputParser
from vlm.hallucination_control import HallucinationDetector, HallucinationPreventer


class TestDefectTokenizer(unittest.TestCase):
    """Test defect tokenizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tokenizer = DefectTokenizer()
        self.sample_detections = [
            {
                "defect_type": "Short",
                "bbox": [120.0, 340.0, 150.0, 370.0],
                "center": [135.0, 355.0],
                "confidence": 0.92,
                "severity": "High"
            },
            {
                "defect_type": "Spur",
                "bbox": [290.0, 410.0, 320.0, 440.0],
                "center": [305.0, 425.0],
                "confidence": 0.87,
                "severity": "Medium"
            }
        ]
    
    def test_encode_basic(self):
        """Test basic encoding."""
        result = self.tokenizer.encode(self.sample_detections)
        self.assertIn("Detected objects:", result)
        self.assertIn("Short", result)
        self.assertIn("Spur", result)
        self.assertIn("135.0", result)  # Center coordinate
    
    def test_encode_empty(self):
        """Test encoding empty detections."""
        result = self.tokenizer.encode([])
        self.assertIn("None", result)
    
    def test_encode_compact(self):
        """Test compact encoding."""
        result = self.tokenizer.encode_compact(self.sample_detections)
        self.assertIn("Detected defects:", result)
        self.assertIn("Short", result)


class TestOutputParser(unittest.TestCase):
    """Test output parser."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = OutputParser(validate=True)
        self.sample_detections = [
            {
                "defect_type": "Short",
                "bbox": [120.0, 340.0, 150.0, 370.0],
                "center": [135.0, 355.0],
                "confidence": 0.92,
                "severity": "High"
            }
        ]
    
    def test_parse_valid_json(self):
        """Test parsing valid JSON."""
        llm_output = '{"count": 1, "defect_type": "Short", "locations": [[135, 355]], "confidence": 0.92, "severity": "High"}'
        result = self.parser.parse(llm_output, "How many shorts?", self.sample_detections)
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["defect_type"], "Short")
    
    def test_parse_invalid_json(self):
        """Test parsing invalid JSON with fallback."""
        llm_output = "I found 1 short defect at coordinates (135, 355)"
        result = self.parser.parse(llm_output, "How many shorts?", self.sample_detections)
        # Should still extract information
        self.assertIn("count", result)
    
    def test_validate_schema(self):
        """Test schema validation."""
        valid_data = {
            "count": 1,
            "defect_type": "Short",
            "locations": [[135, 355]],
            "confidence": 0.92,
            "severity": "High"
        }
        self.assertTrue(self.parser.validate_schema(valid_data))
        
        invalid_data = {"count": 1}  # Missing fields
        self.assertFalse(self.parser.validate_schema(invalid_data))


class TestHallucinationDetector(unittest.TestCase):
    """Test hallucination detector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = HallucinationDetector()
        self.sample_detections = [
            {
                "defect_type": "Short",
                "center": [135.0, 355.0],
                "confidence": 0.92
            }
        ]
    
    def test_detect_non_existent_defect(self):
        """Test detection of non-existent defect."""
        predicted = {
            "count": 1,
            "defect_type": "Crack",  # Not in detections
            "locations": [[100, 200]],
            "confidence": 0.8
        }
        flags = self.detector.detect(predicted, self.sample_detections, "How many cracks?")
        self.assertTrue(flags["non_existent_defect"] or flags["invalid_defect_type"])
    
    def test_detect_count_mismatch(self):
        """Test detection of count mismatch."""
        predicted = {
            "count": 5,  # Wrong count
            "defect_type": "Short",
            "locations": [[135, 355]],
            "confidence": 0.92
        }
        flags = self.detector.detect(predicted, self.sample_detections, "How many shorts?")
        self.assertTrue(flags["count_mismatch"])
    
    def test_no_hallucination(self):
        """Test correct prediction (no hallucination)."""
        predicted = {
            "count": 1,
            "defect_type": "Short",
            "locations": [[135, 355]],
            "confidence": 0.92
        }
        flags = self.detector.detect(predicted, self.sample_detections, "How many shorts?")
        self.assertFalse(self.detector.is_hallucination(flags))


class TestHallucinationPreventer(unittest.TestCase):
    """Test hallucination preventer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preventer = HallucinationPreventer()
        self.sample_detections = [
            {
                "defect_type": "Short",
                "center": [135.0, 355.0],
                "confidence": 0.92
            }
        ]
    
    def test_sanitize_output(self):
        """Test output sanitization."""
        output = {
            "count": 5,  # Wrong
            "defect_type": "Crack",  # Non-existent
            "locations": [[100, 200]],
            "confidence": 0.8
        }
        sanitized = self.preventer.sanitize_output(output, self.sample_detections)
        self.assertEqual(sanitized["count"], 1)  # Corrected
        self.assertEqual(sanitized["defect_type"], "Short")  # Corrected
        self.assertEqual(sanitized["locations"], [[135.0, 355.0]])  # Corrected


if __name__ == "__main__":
    unittest.main()
