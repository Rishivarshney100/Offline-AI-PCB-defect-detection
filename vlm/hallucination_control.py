"""
Hallucination Control: Mechanisms to prevent and detect hallucinations
"""

from typing import Dict, List, Optional, Set
import re


class HallucinationDetector:
    """
    Detects potential hallucinations in LLM outputs.
    """
    
    def __init__(self):
        """Initialize hallucination detector."""
        self.known_defect_types = {
            "missing_hole", "mouse_bite", "open_circuit",
            "short", "spur", "spurious_copper"
        }
    
    def detect(self, llm_output: Dict, detections: List[Dict], 
              query: str) -> Dict[str, bool]:
        """
        Detect potential hallucinations in LLM output.
        
        Args:
            llm_output: Parsed LLM output
            detections: Actual detection results
            query: Original query
        
        Returns:
            Dictionary with hallucination flags
        """
        flags = {
            "non_existent_defect": False,
            "count_mismatch": False,
            "location_mismatch": False,
            "invalid_defect_type": False,
            "hallucinated_confidence": False
        }
        
        # Check for non-existent defect types
        output_type = llm_output.get("defect_type")
        if output_type:
            output_type_norm = output_type.lower().replace(" ", "_")
            detected_types = {d.get("defect_type", "").lower().replace(" ", "_") 
                            for d in detections}
            
            if output_type_norm not in detected_types and output_type_norm not in self.known_defect_types:
                flags["invalid_defect_type"] = True
            
            if output_type_norm not in detected_types and len(detections) > 0:
                flags["non_existent_defect"] = True
        
        # Check count mismatch
        output_count = llm_output.get("count", 0)
        actual_count = len(detections)
        
        if output_count != actual_count:
            flags["count_mismatch"] = True
        
        # Check location mismatch
        output_locations = llm_output.get("locations", [])
        if output_locations and detections:
            actual_centers = [tuple(d.get("center", [0, 0])) for d in detections]
            output_locations_tuples = [tuple(loc) for loc in output_locations]
            
            # Check if locations match (with tolerance)
            if not self._locations_match(output_locations_tuples, actual_centers, tolerance=10.0):
                flags["location_mismatch"] = True
        
        # Check confidence
        output_conf = llm_output.get("confidence", 0.0)
        if output_conf > 0 and not detections:
            flags["hallucinated_confidence"] = True
        
        return flags
    
    def _locations_match(self, predicted: List[tuple], actual: List[tuple], 
                        tolerance: float = 10.0) -> bool:
        """
        Check if predicted locations match actual locations.
        
        Args:
            predicted: List of (x, y) tuples
            actual: List of (x, y) tuples
            tolerance: Distance tolerance in pixels
        
        Returns:
            True if locations match within tolerance
        """
        if len(predicted) != len(actual):
            return False
        
        # Check each predicted location against actual
        matched = set()
        for pred_loc in predicted:
            found_match = False
            for i, actual_loc in enumerate(actual):
                if i in matched:
                    continue
                
                # Calculate distance
                dist = ((pred_loc[0] - actual_loc[0])**2 + 
                       (pred_loc[1] - actual_loc[1])**2)**0.5
                
                if dist <= tolerance:
                    matched.add(i)
                    found_match = True
                    break
            
            if not found_match:
                return False
        
        return True
    
    def is_hallucination(self, flags: Dict[str, bool]) -> bool:
        """
        Determine if output is a hallucination based on flags.
        
        Args:
            flags: Hallucination detection flags
        
        Returns:
            True if hallucination detected
        """
        # Critical flags that indicate hallucination
        critical_flags = [
            "non_existent_defect",
            "invalid_defect_type",
            "hallucinated_confidence"
        ]
        
        return any(flags.get(flag, False) for flag in critical_flags)


class HallucinationPreventer:
    """
    Prevents hallucinations through architectural controls.
    """
    
    def __init__(self):
        """Initialize hallucination preventer."""
        self.known_defect_types = {
            "missing_hole", "mouse_bite", "open_circuit",
            "short", "spur", "spurious_copper",
            "Missing Hole", "Mouse Bite", "Open Circuit",
            "Short", "Spur", "Spurious Copper"
        }
    
    def validate_defect_type(self, defect_type: str, 
                            detections: List[Dict]) -> Optional[str]:
        """
        Validate and correct defect type against detections.
        
        Args:
            defect_type: Defect type from LLM output
            detections: Actual detections
        
        Returns:
            Validated defect type or None if invalid
        """
        if not defect_type:
            return None
        
        # Normalize
        defect_type_norm = defect_type.strip()
        
        # Check if it's a known type
        if defect_type_norm not in self.known_defect_types:
            return None
        
        # Check if it exists in detections
        detected_types = {d.get("defect_type", "") for d in detections}
        if defect_type_norm not in detected_types and len(detections) > 0:
            # Use the actual type from detections
            if detected_types:
                return list(detected_types)[0]
            return None
        
        return defect_type_norm
    
    def validate_count(self, predicted_count: int, 
                      detections: List[Dict], 
                      defect_type: Optional[str] = None) -> int:
        """
        Validate and correct count against detections.
        
        Args:
            predicted_count: Count from LLM output
            detections: Actual detections
            defect_type: Optional defect type filter
        
        Returns:
            Corrected count
        """
        if defect_type:
            # Filter by defect type
            filtered = [
                d for d in detections 
                if d.get("defect_type", "").lower().replace(" ", "_") == 
                   defect_type.lower().replace(" ", "_")
            ]
            return len(filtered)
        else:
            return len(detections)
    
    def validate_locations(self, predicted_locations: List[List[float]],
                          detections: List[Dict]) -> List[List[float]]:
        """
        Validate and correct locations against detection centers.
        
        Args:
            predicted_locations: Locations from LLM output
            detections: Actual detections
        
        Returns:
            Corrected locations (actual centers)
        """
        if not detections:
            return []
        
        # Use actual centers from detections
        actual_locations = [list(d.get("center", [0, 0])) for d in detections]
        return actual_locations
    
    def sanitize_output(self, output: Dict, detections: List[Dict]) -> Dict:
        """
        Sanitize LLM output to prevent hallucinations.
        
        Args:
            output: LLM output dictionary
            detections: Actual detections
        
        Returns:
            Sanitized output
        """
        sanitized = output.copy()
        
        # Validate defect type
        defect_type = sanitized.get("defect_type")
        if defect_type:
            validated_type = self.validate_defect_type(defect_type, detections)
            sanitized["defect_type"] = validated_type
        
        # Validate count
        count = sanitized.get("count", 0)
        validated_count = self.validate_count(count, detections, defect_type)
        sanitized["count"] = validated_count
        
        # Validate locations
        locations = sanitized.get("locations", [])
        validated_locations = self.validate_locations(locations, detections)
        sanitized["locations"] = validated_locations
        
        # If count is 0, clear other fields
        if validated_count == 0:
            sanitized["defect_type"] = None
            sanitized["locations"] = []
            sanitized["confidence"] = 0.0
            sanitized["severity"] = None
        
        return sanitized


class HallucinationMetrics:
    """
    Tracks hallucination metrics for evaluation.
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.total_queries = 0
        self.hallucination_count = 0
        self.hallucination_types = {
            "non_existent_defect": 0,
            "count_mismatch": 0,
            "location_mismatch": 0,
            "invalid_defect_type": 0
        }
    
    def record(self, is_hallucination: bool, flags: Optional[Dict] = None):
        """
        Record hallucination event.
        
        Args:
            is_hallucination: Whether hallucination occurred
            flags: Hallucination flags
        """
        self.total_queries += 1
        
        if is_hallucination:
            self.hallucination_count += 1
            
            if flags:
                for flag_type, value in flags.items():
                    if value and flag_type in self.hallucination_types:
                        self.hallucination_types[flag_type] += 1
    
    def get_rate(self) -> float:
        """
        Get hallucination rate.
        
        Returns:
            Hallucination rate (0.0 to 1.0)
        """
        if self.total_queries == 0:
            return 0.0
        return self.hallucination_count / self.total_queries
    
    def get_report(self) -> Dict:
        """
        Get metrics report.
        
        Returns:
            Dictionary with metrics
        """
        return {
            "total_queries": self.total_queries,
            "hallucination_count": self.hallucination_count,
            "hallucination_rate": self.get_rate(),
            "by_type": self.hallucination_types.copy()
        }
    
    def reset(self):
        """Reset metrics."""
        self.total_queries = 0
        self.hallucination_count = 0
        self.hallucination_types = {k: 0 for k in self.hallucination_types}
