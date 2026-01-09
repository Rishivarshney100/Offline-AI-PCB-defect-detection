"""
Response Generator for creating natural language responses
about PCB defects
Supports both rule-based and VLM-based generation
"""

from typing import Dict, List, Optional
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class ResponseGenerator:
    """
    Generates natural language responses based on detection results
    and query intent.
    Supports both rule-based (legacy) and VLM-based generation.
    """
    
    def __init__(self, llm_config: Optional[Dict] = None):
        """
        Initialize the response generator.
        
        Args:
            llm_config: Configuration for language model
                - type: "rule_based" (default) or "vlm"
                - vlm_config: VLM configuration if type is "vlm"
        """
        self.llm_config = llm_config or {}
        self.response_type = self.llm_config.get("type", "rule_based")
        
        # Initialize VLM if configured
        self.vlm = None
        if self.response_type == "vlm":
            try:
                from vlm.vlm_model import DetectionGroundedVLM
                from vlm.architecture import VLMConfig
                
                vlm_config_dict = self.llm_config.get("vlm_config", {})
                vlm_config = VLMConfig(**vlm_config_dict)
                self.vlm = DetectionGroundedVLM(config=vlm_config)
                print("VLM initialized successfully")
            except Exception as e:
                print(f"Warning: VLM initialization failed: {e}")
                print("Falling back to rule-based generation")
                self.response_type = "rule_based"
    
    def generate_response(self, query: str, query_intent: Dict, 
                         detection_results: Dict, image_path: Optional[str] = None) -> str:
        """
        Generate a natural language response based on query and results.
        
        Args:
            query: Original natural language query
            query_intent: Processed query intent from QueryProcessor
            detection_results: Results from defect detection
            image_path: Path to image (required for VLM mode)
            
        Returns:
            Natural language response string
        """
        # Use VLM if available and image path provided
        if self.response_type == "vlm" and self.vlm is not None and image_path:
            try:
                return self.vlm.generate_natural_language(image_path, query)
            except Exception as e:
                print(f"Warning: VLM generation failed: {e}")
                print("Falling back to rule-based generation")
                # Fall through to rule-based
        
        # Rule-based generation (legacy)
        intent_type = query_intent["type"]
        defects = detection_results.get("defects", [])
        total_defects = detection_results.get("total_defects", 0)
        
        if total_defects == 0:
            return self._generate_no_defects_response(query_intent)
        
        # Route to appropriate response generator based on intent
        if intent_type == "count":
            return self._generate_count_response(defects, total_defects, query_intent)
        elif intent_type == "type":
            return self._generate_type_response(defects, query_intent)
        elif intent_type == "location":
            return self._generate_location_response(defects, query_intent)
        elif intent_type == "severity":
            return self._generate_severity_response(defects, query_intent)
        elif intent_type == "specific_defect":
            return self._generate_specific_defect_response(defects, query_intent)
        elif intent_type == "confidence":
            return self._generate_confidence_response(defects, query_intent)
        else:
            return self._generate_general_response(defects, total_defects, query_intent)
    
    def generate_structured_response(self, image_path: str, query: str) -> Dict:
        """
        Generate structured JSON response (VLM only).
        
        Args:
            image_path: Path to PCB image
            query: Natural language query
            
        Returns:
            Structured response dictionary
        """
        if self.response_type == "vlm" and self.vlm is not None:
            return self.vlm.generate(image_path, query)
        else:
            raise ValueError("Structured response requires VLM mode")
    
    def _generate_no_defects_response(self, query_intent: Dict) -> str:
        """Generate response when no defects are detected."""
        return "Great news! No defects were detected in this PCB image. The board appears to be defect-free."
    
    def _generate_count_response(self, defects: List[Dict], total: int, 
                                 query_intent: Dict) -> str:
        """Generate response for count queries."""
        if query_intent.get("defect_type"):
            # Count specific defect type
            defect_type = query_intent["defect_type"]
            count = sum(1 for d in defects if d["defect_type"].lower().replace(" ", "_") == defect_type)
            defect_name = defect_type.replace("_", " ").title()
            return f"I found {count} {defect_name} defect(s) in this PCB image."
        else:
            return f"I detected {total} defect(s) in total in this PCB image."
    
    def _generate_type_response(self, defects: List[Dict], query_intent: Dict) -> str:
        """Generate response for defect type queries."""
        defect_types = {}
        for defect in defects:
            d_type = defect["defect_type"]
            defect_types[d_type] = defect_types.get(d_type, 0) + 1
        
        if len(defect_types) == 0:
            return "No defects were found."
        
        type_list = [f"{count} {dtype}" for dtype, count in defect_types.items()]
        if len(type_list) == 1:
            return f"I found {type_list[0]} defect(s) in this image."
        else:
            types_str = ", ".join(type_list[:-1]) + f", and {type_list[-1]}"
            return f"I found the following defect types: {types_str}."
    
    def _generate_location_response(self, defects: List[Dict], query_intent: Dict) -> str:
        """Generate response for location queries."""
        if query_intent.get("defect_type"):
            # Location of specific defect type
            defect_type = query_intent["defect_type"]
            matching_defects = [d for d in defects 
                              if d["defect_type"].lower().replace(" ", "_") == defect_type]
        else:
            matching_defects = defects
        
        if not matching_defects:
            return "No matching defects found for location information."
        
        locations = []
        for i, defect in enumerate(matching_defects, 1):
            center = defect["center"]
            locations.append(f"Defect {i} ({defect['defect_type']}) is located at coordinates ({center[0]:.1f}, {center[1]:.1f})")
        
        return "Here are the defect locations:\n" + "\n".join(locations)
    
    def _generate_severity_response(self, defects: List[Dict], query_intent: Dict) -> str:
        """Generate response for severity queries."""
        severity_counts = {"Low": 0, "Medium": 0, "High": 0}
        for defect in defects:
            severity = defect.get("severity", "Low")
            if severity in severity_counts:
                severity_counts[severity] += 1
        
        if query_intent.get("parameters"):
            # Filter by specific severity
            if "high_severity" in query_intent["parameters"]:
                return f"I found {severity_counts['High']} high severity defect(s) that require immediate attention."
            elif "low_severity" in query_intent["parameters"]:
                return f"I found {severity_counts['Low']} low severity defect(s)."
            elif "medium_severity" in query_intent["parameters"]:
                return f"I found {severity_counts['Medium']} medium severity defect(s)."
        
        # General severity summary
        severity_info = []
        for sev, count in severity_counts.items():
            if count > 0:
                severity_info.append(f"{count} {sev.lower()} severity")
        
        if not severity_info:
            return "No defects found."
        
        return f"Severity distribution: {', '.join(severity_info)} defect(s)."
    
    def _generate_specific_defect_response(self, defects: List[Dict], query_intent: Dict) -> str:
        """Generate response for specific defect type queries."""
        defect_type = query_intent.get("defect_type")
        if not defect_type:
            return self._generate_type_response(defects, query_intent)
        
        matching_defects = [d for d in defects 
                          if d["defect_type"].lower().replace(" ", "_") == defect_type]
        
        if not matching_defects:
            defect_name = defect_type.replace("_", " ").title()
            return f"No {defect_name} defects were found in this image."
        
        defect_name = defect_type.replace("_", " ").title()
        count = len(matching_defects)
        response = f"I found {count} {defect_name} defect(s). "
        
        details = []
        for i, defect in enumerate(matching_defects, 1):
            details.append(f"Defect {i}: confidence {defect['confidence']:.2%}, severity {defect['severity']}")
        
        return response + "Details: " + "; ".join(details)
    
    def _generate_confidence_response(self, defects: List[Dict], query_intent: Dict) -> str:
        """Generate response for confidence queries."""
        if not defects:
            return "No defects detected."
        
        avg_confidence = sum(d["confidence"] for d in defects) / len(defects)
        max_confidence = max(d["confidence"] for d in defects)
        min_confidence = min(d["confidence"] for d in defects)
        
        return (f"Confidence scores: average {avg_confidence:.2%}, "
                f"highest {max_confidence:.2%}, lowest {min_confidence:.2%}.")
    
    def _generate_general_response(self, defects: List[Dict], total: int, 
                                   query_intent: Dict) -> str:
        """Generate general summary response."""
        if total == 0:
            return "No defects were detected in this PCB image."
        
        # Get defect type summary
        defect_types = {}
        severity_counts = {"Low": 0, "Medium": 0, "High": 0}
        
        for defect in defects:
            d_type = defect["defect_type"]
            defect_types[d_type] = defect_types.get(d_type, 0) + 1
            severity = defect.get("severity", "Low")
            if severity in severity_counts:
                severity_counts[severity] += 1
        
        response = f"I analyzed the PCB image and found {total} defect(s). "
        
        if len(defect_types) > 0:
            type_list = list(defect_types.keys())
            if len(type_list) == 1:
                response += f"The defect type is {type_list[0]}. "
            else:
                response += f"Defect types include: {', '.join(type_list)}. "
        
        # Add severity info
        high_severity = severity_counts.get("High", 0)
        if high_severity > 0:
            response += f"Warning: {high_severity} high severity defect(s) detected that require attention. "
        
        return response.strip()
