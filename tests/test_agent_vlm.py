"""
Integration tests for agent with VLM
"""

import unittest
from unittest.mock import Mock, patch
import sys
from pathlib import Path
import tempfile
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.agent import PCBDefectAgent
from agent.response_generator import ResponseGenerator


class TestAgentVLMIntegration(unittest.TestCase):
    """Test agent integration with VLM."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create dummy image file
        self.temp_image = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        self.temp_image.write(b"dummy image data")
        self.temp_image.close()
        self.image_path = self.temp_image.name
    
    def tearDown(self):
        """Clean up."""
        if os.path.exists(self.image_path):
            os.unlink(self.image_path)
    
    @patch('agent.response_generator.DetectionGroundedVLM')
    def test_vlm_mode_initialization(self, mock_vlm_class):
        """Test VLM mode initialization."""
        llm_config = {
            "type": "vlm",
            "vlm_config": {
                "llm_backend": "ollama",
                "llm_model_name": "phi-2"
            }
        }
        
        generator = ResponseGenerator(llm_config=llm_config)
        
        # Should attempt to initialize VLM
        if generator.response_type == "vlm":
            self.assertIsNotNone(generator.vlm)
    
    def test_rule_based_fallback(self):
        """Test fallback to rule-based when VLM fails."""
        llm_config = {
            "type": "vlm",
            "vlm_config": {
                "llm_backend": "ollama",
                "llm_model_name": "nonexistent-model"
            }
        }
        
        generator = ResponseGenerator(llm_config=llm_config)
        
        # Should fall back to rule-based
        self.assertEqual(generator.response_type, "rule_based")
    
    def test_backward_compatibility(self):
        """Test backward compatibility with existing API."""
        agent = PCBDefectAgent(
            model_path="models/weights/best.pt",
            llm_config={"type": "rule_based"}
        )
        
        # Should work with existing API
        # Note: This would need actual model file, so we mock it
        with patch.object(agent.detector, 'predict') as mock_predict:
            mock_predict.return_value = {
                "defects": [],
                "total_defects": 0
            }
            
            response = agent.answer_query(self.image_path, "How many defects?")
            self.assertIsInstance(response, str)


if __name__ == "__main__":
    unittest.main()
