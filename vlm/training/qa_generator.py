"""
Synthetic QA Pair Generator: Creates training data from bounding box annotations
"""

from typing import List, Dict, Optional, Tuple
import random
import json
from pathlib import Path


class QAGenerator:
    """
    Generates synthetic question-answer pairs from detection annotations.
    """
    
    # Question templates
    COUNT_TEMPLATES = [
        "How many {defect_type} defects are present?",
        "What is the count of {defect_type} defects?",
        "How many {defect_type} are there?",
        "Count the {defect_type} defects.",
        "Number of {defect_type} defects?"
    ]
    
    TYPE_TEMPLATES = [
        "What types of defects are present?",
        "What defect types were found?",
        "List all defect types.",
        "Which defects are in this image?",
        "What kinds of defects are there?"
    ]
    
    LOCATION_TEMPLATES = [
        "Where are the {defect_type} defects located?",
        "What are the coordinates of {defect_type} defects?",
        "Locate the {defect_type} defects.",
        "Where can I find {defect_type} defects?",
        "Position of {defect_type} defects?"
    ]
    
    SEVERITY_TEMPLATES = [
        "What is the severity of the defects?",
        "How severe are the defects?",
        "Which defects are high severity?",
        "List high severity defects.",
        "What are the most critical defects?"
    ]
    
    CONFIDENCE_TEMPLATES = [
        "What is the confidence of the detections?",
        "How confident are the defect detections?",
        "What is the average confidence?",
        "Confidence scores of defects?"
    ]
    
    SPECIFIC_TEMPLATES = [
        "Tell me about {defect_type} defects.",
        "Describe the {defect_type} defects.",
        "Information about {defect_type} defects.",
        "Details of {defect_type} defects."
    ]
    
    def __init__(self, seed: int = 42):
        """
        Initialize QA generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
    
    def generate_qa_pairs(self, detections: List[Dict], 
                         image_path: str,
                         num_pairs: int = 10) -> List[Dict]:
        """
        Generate QA pairs from detection results.
        
        Args:
            detections: List of detection dictionaries
            image_path: Path to image
            num_pairs: Number of QA pairs to generate
        
        Returns:
            List of QA pair dictionaries
        """
        qa_pairs = []
        
        if not detections:
            # Generate "no defects" QA pairs
            qa_pairs.extend(self._generate_no_defect_qa(num_pairs // 2))
            return qa_pairs
        
        # Generate different types of questions
        question_types = ["count", "type", "location", "severity", "confidence", "specific"]
        weights = [0.3, 0.2, 0.2, 0.1, 0.1, 0.1]  # Weighted distribution
        
        for _ in range(num_pairs):
            q_type = random.choices(question_types, weights=weights)[0]
            
            if q_type == "count":
                qa_pairs.extend(self._generate_count_qa(detections))
            elif q_type == "type":
                qa_pairs.extend(self._generate_type_qa(detections))
            elif q_type == "location":
                qa_pairs.extend(self._generate_location_qa(detections))
            elif q_type == "severity":
                qa_pairs.extend(self._generate_severity_qa(detections))
            elif q_type == "confidence":
                qa_pairs.extend(self._generate_confidence_qa(detections))
            elif q_type == "specific":
                qa_pairs.extend(self._generate_specific_qa(detections))
        
        # Limit to requested number
        return qa_pairs[:num_pairs]
    
    def _generate_count_qa(self, detections: List[Dict]) -> List[Dict]:
        """Generate counting questions."""
        qa_pairs = []
        
        # Total count
        total_count = len(detections)
        question = random.choice(self.COUNT_TEMPLATES).format(defect_type="defect")
        answer = {
            "count": total_count,
            "defect_type": None,
            "locations": [],
            "confidence": 0.0,
            "severity": None
        }
        qa_pairs.append({
            "question": question,
            "answer": answer,
            "grounding": list(range(len(detections)))
        })
        
        # Count by type
        by_type = {}
        for i, det in enumerate(detections):
            defect_type = det.get("defect_type", "Unknown")
            if defect_type not in by_type:
                by_type[defect_type] = []
            by_type[defect_type].append(i)
        
        for defect_type, indices in by_type.items():
            question = random.choice(self.COUNT_TEMPLATES).format(
                defect_type=defect_type.lower()
            )
            answer = {
                "count": len(indices),
                "defect_type": defect_type,
                "locations": [detections[i].get("center", [0, 0]) for i in indices],
                "confidence": sum(detections[i].get("confidence", 0) for i in indices) / len(indices),
                "severity": self._get_most_common_severity([detections[i] for i in indices])
            }
            qa_pairs.append({
                "question": question,
                "answer": answer,
                "grounding": indices
            })
        
        return qa_pairs
    
    def _generate_type_qa(self, detections: List[Dict]) -> List[Dict]:
        """Generate defect type questions."""
        qa_pairs = []
        
        defect_types = list(set(d.get("defect_type", "Unknown") for d in detections))
        
        question = random.choice(self.TYPE_TEMPLATES)
        answer = {
            "count": len(detections),
            "defect_type": defect_types[0] if defect_types else None,
            "locations": [d.get("center", [0, 0]) for d in detections],
            "confidence": sum(d.get("confidence", 0) for d in detections) / len(detections) if detections else 0.0,
            "severity": self._get_most_common_severity(detections)
        }
        
        qa_pairs.append({
            "question": question,
            "answer": answer,
            "grounding": list(range(len(detections)))
        })
        
        return qa_pairs
    
    def _generate_location_qa(self, detections: List[Dict]) -> List[Dict]:
        """Generate location questions."""
        qa_pairs = []
        
        # Location of all defects
        question = "Where are the defects located?"
        answer = {
            "count": len(detections),
            "defect_type": None,
            "locations": [d.get("center", [0, 0]) for d in detections],
            "confidence": 0.0,
            "severity": None
        }
        qa_pairs.append({
            "question": question,
            "answer": answer,
            "grounding": list(range(len(detections)))
        })
        
        # Location by type
        by_type = {}
        for i, det in enumerate(detections):
            defect_type = det.get("defect_type", "Unknown")
            if defect_type not in by_type:
                by_type[defect_type] = []
            by_type[defect_type].append(i)
        
        for defect_type, indices in by_type.items():
            question = random.choice(self.LOCATION_TEMPLATES).format(
                defect_type=defect_type.lower()
            )
            answer = {
                "count": len(indices),
                "defect_type": defect_type,
                "locations": [detections[i].get("center", [0, 0]) for i in indices],
                "confidence": sum(detections[i].get("confidence", 0) for i in indices) / len(indices),
                "severity": self._get_most_common_severity([detections[i] for i in indices])
            }
            qa_pairs.append({
                "question": question,
                "answer": answer,
                "grounding": indices
            })
        
        return qa_pairs
    
    def _generate_severity_qa(self, detections: List[Dict]) -> List[Dict]:
        """Generate severity questions."""
        qa_pairs = []
        
        question = random.choice(self.SEVERITY_TEMPLATES)
        
        # Count by severity
        by_severity = {"Low": [], "Medium": [], "High": []}
        for i, det in enumerate(detections):
            severity = det.get("severity", "Low")
            if severity in by_severity:
                by_severity[severity].append(i)
        
        # Find highest severity
        highest_severity = None
        if by_severity["High"]:
            highest_severity = "High"
        elif by_severity["Medium"]:
            highest_severity = "Medium"
        elif by_severity["Low"]:
            highest_severity = "Low"
        
        answer = {
            "count": len(detections),
            "defect_type": None,
            "locations": [d.get("center", [0, 0]) for d in detections],
            "confidence": 0.0,
            "severity": highest_severity
        }
        
        qa_pairs.append({
            "question": question,
            "answer": answer,
            "grounding": list(range(len(detections)))
        })
        
        return qa_pairs
    
    def _generate_confidence_qa(self, detections: List[Dict]) -> List[Dict]:
        """Generate confidence questions."""
        qa_pairs = []
        
        question = random.choice(self.CONFIDENCE_TEMPLATES)
        
        confidences = [d.get("confidence", 0.0) for d in detections]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        answer = {
            "count": len(detections),
            "defect_type": None,
            "locations": [],
            "confidence": avg_confidence,
            "severity": None
        }
        
        qa_pairs.append({
            "question": question,
            "answer": answer,
            "grounding": list(range(len(detections)))
        })
        
        return qa_pairs
    
    def _generate_specific_qa(self, detections: List[Dict]) -> List[Dict]:
        """Generate specific defect type questions."""
        qa_pairs = []
        
        # Group by type
        by_type = {}
        for i, det in enumerate(detections):
            defect_type = det.get("defect_type", "Unknown")
            if defect_type not in by_type:
                by_type[defect_type] = []
            by_type[defect_type].append(i)
        
        for defect_type, indices in by_type.items():
            question = random.choice(self.SPECIFIC_TEMPLATES).format(
                defect_type=defect_type.lower()
            )
            answer = {
                "count": len(indices),
                "defect_type": defect_type,
                "locations": [detections[i].get("center", [0, 0]) for i in indices],
                "confidence": sum(detections[i].get("confidence", 0) for i in indices) / len(indices),
                "severity": self._get_most_common_severity([detections[i] for i in indices])
            }
            qa_pairs.append({
                "question": question,
                "answer": answer,
                "grounding": indices
            })
        
        return qa_pairs
    
    def _generate_no_defect_qa(self, num: int) -> List[Dict]:
        """Generate QA pairs for images with no defects."""
        qa_pairs = []
        
        questions = [
            "How many defects are present?",
            "What defects are in this image?",
            "Are there any defects?",
            "Count the defects.",
            "What types of defects are there?"
        ]
        
        answer = {
            "count": 0,
            "defect_type": None,
            "locations": [],
            "confidence": 0.0,
            "severity": None
        }
        
        for _ in range(num):
            question = random.choice(questions)
            qa_pairs.append({
                "question": question,
                "answer": answer,
                "grounding": []
            })
        
        return qa_pairs
    
    def _get_most_common_severity(self, detections: List[Dict]) -> Optional[str]:
        """Get most common severity from detections."""
        if not detections:
            return None
        
        severities = [d.get("severity", "Low") for d in detections]
        from collections import Counter
        most_common = Counter(severities).most_common(1)
        return most_common[0][0] if most_common else "Low"
    
    def generate_from_image(self, image_path: str, 
                           detector,
                           num_pairs: int = 10) -> List[Dict]:
        """
        Generate QA pairs from image using detector.
        
        Args:
            image_path: Path to image
            detector: PCBDefectDetector instance
            num_pairs: Number of QA pairs to generate
        
        Returns:
            List of QA pair dictionaries
        """
        # Detect defects
        detection_results = detector.predict(image_path)
        detections = detection_results.get("defects", [])
        
        # Generate QA pairs
        qa_pairs = self.generate_qa_pairs(detections, image_path, num_pairs)
        
        return qa_pairs
    
    def save_qa_dataset(self, qa_pairs: List[Dict], output_path: str):
        """
        Save QA pairs to JSON file.
        
        Args:
            qa_pairs: List of QA pair dictionaries
            output_path: Path to save JSON file
        """
        dataset = {
            "version": "1.0",
            "qa_pairs": qa_pairs,
            "total_pairs": len(qa_pairs)
        }
        
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"Saved {len(qa_pairs)} QA pairs to {output_path}")
