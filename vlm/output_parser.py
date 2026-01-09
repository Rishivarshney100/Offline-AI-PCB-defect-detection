"""
Output Parser: Parses and validates LLM output against detection results
"""

from typing import Dict, List, Optional, Any
import json
import re


class OutputParser:
    """
    Parses LLM JSON output and validates it against detection results
    to prevent hallucination.
    """
    
    def __init__(self, validate: bool = True):
        """
        Initialize output parser.
        
        Args:
            validate: Whether to validate output against detections
        """
        self.validate = validate
    
    def parse(self, 
              llm_output: str, 
              query: str,
              detections: List[Dict]) -> Dict:
        """
        Parse LLM output and validate against detections.
        
        Args:
            llm_output: Raw output from LLM
            query: Original user query
            detections: List of detection results for validation
        
        Returns:
            Parsed and validated response dictionary
        """
        # Extract JSON from LLM output
        json_str = self._extract_json(llm_output)
        
        # Parse JSON
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            # Fallback: try to extract structured info
            parsed = self._parse_fallback(llm_output, detections)
        
        # Validate against detections
        if self.validate:
            parsed = self._validate(parsed, query, detections)
        
        return parsed
    
    def _extract_json(self, text: str) -> str:
        """
        Extract JSON from LLM output text.
        
        Args:
            text: Raw LLM output
        
        Returns:
            JSON string
        """
        # Try to find JSON block
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        # Try code block
        code_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if code_match:
            return code_match.group(1)
        
        # Return as-is if no JSON found
        return text.strip()
    
    def _parse_fallback(self, text: str, detections: List[Dict]) -> Dict:
        result = {"count": 0, "defect_type": None, "locations": [], "confidence": 0.0, "severity": None}
        if m := re.search(r'"count"\s*:\s*(\d+)', text):
            result["count"] = int(m.group(1))
        if m := re.search(r'"defect_type"\s*:\s*"([^"]+)"', text):
            result["defect_type"] = m.group(1)
        if m := re.search(r'"locations"\s*:\s*\[(.*?)\]', text, re.DOTALL):
            result["locations"] = [[float(x), float(y)] for x, y in re.findall(r'\[(\d+\.?\d*),\s*(\d+\.?\d*)\]', m.group(1))]
        if m := re.search(r'"confidence"\s*:\s*([\d.]+)', text):
            result["confidence"] = float(m.group(1))
        return result
    
    def _validate(self, parsed: Dict, query: str, detections: List[Dict]) -> Dict:
        ql = query.lower()
        defect_types = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
        query_type = next((dt for dt in defect_types if dt.replace("_", " ") in ql), None)
        relevant = [d for d in detections if not query_type or d.get("defect_type", "").lower().replace(" ", "_") == query_type] if query_type else detections
        
        if parsed.get("count", 0) != len(relevant):
            parsed["count"] = len(relevant)
        
        pred_type = parsed.get("defect_type")
        if pred_type:
            pred_norm = pred_type.lower().replace(" ", "_")
            actual_types = {d.get("defect_type", "").lower().replace(" ", "_") for d in relevant}
            if pred_norm not in actual_types and actual_types:
                parsed["defect_type"] = relevant[0].get("defect_type")
        
        if parsed.get("locations") and relevant:
            parsed["locations"] = [d.get("center", [0, 0]) for d in relevant]
        
        if relevant:
            confs = [d.get("confidence", 0.0) for d in relevant]
            if not parsed.get("confidence"):
                parsed["confidence"] = sum(confs) / len(confs)
            sevs = [d.get("severity", "Low") for d in relevant]
            if not parsed.get("severity"):
                from collections import Counter
                parsed["severity"] = Counter(sevs).most_common(1)[0][0]
        
        if parsed["count"] == 0:
            parsed.update({"defect_type": None, "locations": [], "confidence": 0.0, "severity": None})
        return parsed
    
    def validate_schema(self, data: Dict) -> bool:
        required = ["count", "defect_type", "locations", "confidence", "severity"]
        if not all(f in data for f in required):
            return False
        return (isinstance(data["count"], int) and
                (data["defect_type"] is None or isinstance(data["defect_type"], str)) and
                isinstance(data["locations"], list) and
                isinstance(data["confidence"], (int, float)) and
                (data["severity"] is None or data["severity"] in ["Low", "Medium", "High"]))
