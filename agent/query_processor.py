"""
Query Processor for understanding natural language queries
about PCB defects
"""

from typing import Dict, List
import re


class QueryProcessor:
    """
    Processes natural language queries to extract intent and parameters.
    """
    
    # Query patterns and intents
    QUERY_PATTERNS = {
        "count": [
            r"how many",
            r"number of",
            r"count",
            r"total",
            r"how many defects"
        ],
        "type": [
            r"what (kind|type|types) of",
            r"which defects",
            r"defect types",
            r"what defects"
        ],
        "location": [
            r"where",
            r"location",
            r"position",
            r"coordinates",
            r"center"
        ],
        "severity": [
            r"severity",
            r"how severe",
            r"critical",
            r"serious",
            r"how bad"
        ],
        "specific_defect": [
            r"missing hole",
            r"mouse bite",
            r"open circuit",
            r"short",
            r"spur",
            r"spurious copper"
        ],
        "confidence": [
            r"confidence",
            r"how sure",
            r"certainty",
            r"accuracy"
        ],
        "general": [
            r"tell me",
            r"describe",
            r"explain",
            r"what",
            r"analyze",
            r"summary"
        ]
    }
    
    DEFECT_KEYWORDS = {
        "missing_hole": ["missing hole", "absent hole", "hole missing"],
        "mouse_bite": ["mouse bite", "edge defect", "incomplete cut"],
        "open_circuit": ["open circuit", "broken circuit", "circuit break"],
        "short": ["short", "short circuit", "connection"],
        "spur": ["spur", "unwanted trace", "extra trace"],
        "spurious_copper": ["spurious copper", "excess copper", "extra copper"]
    }
    
    def __init__(self):
        """Initialize the query processor."""
        pass
    
    def process_query(self, query: str) -> Dict:
        """
        Process a natural language query to extract intent.
        
        Args:
            query: Natural language query string
            
        Returns:
            Dictionary with query intent and extracted information
        """
        query_lower = query.lower()
        
        intent = {
            "type": "general",  # count, type, location, severity, specific_defect, confidence, general
            "defect_type": None,
            "parameters": []
        }
        
        # Determine intent type
        for intent_type, patterns in self.QUERY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    intent["type"] = intent_type
                    break
            if intent["type"] != "general":
                break
        
        # Extract specific defect type if mentioned
        for defect_key, keywords in self.DEFECT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    intent["defect_type"] = defect_key
                    break
            if intent["defect_type"]:
                break
        
        # Extract additional parameters
        if "severity" in query_lower or "severe" in query_lower:
            if "high" in query_lower or "critical" in query_lower:
                intent["parameters"].append("high_severity")
            elif "low" in query_lower or "minor" in query_lower:
                intent["parameters"].append("low_severity")
            elif "medium" in query_lower:
                intent["parameters"].append("medium_severity")
        
        return intent
