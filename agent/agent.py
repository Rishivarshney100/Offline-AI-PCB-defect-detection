"""
Main AI Agent for PCB Defect Analysis
Integrates defect detection with natural language processing
"""

from typing import Dict, List, Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.infer import PCBDefectDetector
from agent.query_processor import QueryProcessor
from agent.response_generator import ResponseGenerator


class PCBDefectAgent:
    """
    Main AI Agent that processes images and answers natural language queries
    about PCB defects.
    """
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25, 
                 device: str = "", llm_config: Optional[Dict] = None):
        """
        Initialize the AI agent.
        
        Args:
            model_path: Path to trained YOLOv5 model
            conf_threshold: Confidence threshold for defect detection
            device: Device to use for inference
            llm_config: Configuration for the language model
        """
        # Initialize defect detector
        self.detector = PCBDefectDetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            device=device
        )
        
        # Initialize query processor and response generator
        self.query_processor = QueryProcessor()
        self.response_generator = ResponseGenerator(llm_config=llm_config)
    
    def analyze_image(self, image_path: str) -> Dict:
        """
        Analyze a PCB image and detect defects.
        
        Args:
            image_path: Path to the PCB image
            
        Returns:
            Dictionary with defect detection results
        """
        return self.detector.predict(image_path)
    
    def answer_query(self, image_path: str, query: str) -> str:
        """
        Answer a natural language query about defects in an image.
        
        Args:
            image_path: Path to the PCB image
            query: Natural language query about the defects
            
        Returns:
            Natural language response to the query
        """
        # Analyze the image first
        detection_results = self.analyze_image(image_path)
        
        # Process the query to understand intent
        query_intent = self.query_processor.process_query(query)
        
        # Generate natural language response
        # Pass image_path for VLM mode
        response = self.response_generator.generate_response(
            query=query,
            query_intent=query_intent,
            detection_results=detection_results,
            image_path=image_path
        )
        
        return response
    
    def answer_query_structured(self, image_path: str, query: str) -> Dict:
        """
        Answer query with structured JSON response (VLM only).
        
        Args:
            image_path: Path to the PCB image
            query: Natural language query about the defects
            
        Returns:
            Structured response dictionary with JSON format
        """
        return self.response_generator.generate_structured_response(image_path, query)
    
    def get_defect_summary(self, image_path: str) -> Dict:
        """
        Get a comprehensive summary of defects in an image.
        
        Args:
            image_path: Path to the PCB image
            
        Returns:
            Dictionary with summary information
        """
        results = self.analyze_image(image_path)
        
        summary = {
            "total_defects": results["total_defects"],
            "defect_types": {},
            "severity_distribution": {"Low": 0, "Medium": 0, "High": 0},
            "defects": results["defects"]
        }
        
        # Count defect types and severity
        for defect in results["defects"]:
            defect_type = defect["defect_type"]
            severity = defect["severity"]
            
            if defect_type not in summary["defect_types"]:
                summary["defect_types"][defect_type] = 0
            summary["defect_types"][defect_type] += 1
            
            summary["severity_distribution"][severity] += 1
        
        return summary
