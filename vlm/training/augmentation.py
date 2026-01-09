"""
Data augmentation for QA pairs and training data
"""

from typing import List, Dict
import random


class QAAugmentation:
    """
    Augments QA pairs for training data diversity.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize augmentation.
        
        Args:
            seed: Random seed
        """
        random.seed(seed)
    
    def augment_questions(self, qa_pairs: List[Dict]) -> List[Dict]:
        """
        Augment questions through paraphrasing.
        
        Args:
            qa_pairs: Original QA pairs
        
        Returns:
            Augmented QA pairs
        """
        augmented = []
        
        for qa in qa_pairs:
            # Keep original
            augmented.append(qa)
            
            # Add paraphrased version
            paraphrased = self._paraphrase_question(qa["question"])
            if paraphrased != qa["question"]:
                new_qa = qa.copy()
                new_qa["question"] = paraphrased
                augmented.append(new_qa)
        
        return augmented
    
    def _paraphrase_question(self, question: str) -> str:
        """
        Paraphrase question.
        
        Args:
            question: Original question
        
        Returns:
            Paraphrased question
        """
        # Simple paraphrasing rules
        paraphrases = {
            "how many": ["what is the count of", "number of", "total count of"],
            "where are": ["what are the locations of", "position of", "coordinates of"],
            "what types": ["which types", "what kinds of", "list the types of"],
            "are present": ["are there", "exist", "can be found"],
            "defects": ["defect", "issues", "problems"]
        }
        
        paraphrased = question.lower()
        for original, alternatives in paraphrases.items():
            if original in paraphrased:
                alternative = random.choice(alternatives)
                paraphrased = paraphrased.replace(original, alternative)
                break
        
        return paraphrased.capitalize()
    
    def augment_spatial(self, qa_pairs: List[Dict], 
                       image_width: int, 
                       image_height: int) -> List[Dict]:
        """
        Augment with spatial reasoning questions.
        
        Args:
            qa_pairs: Original QA pairs
            image_width: Image width
            image_height: Image height
        
        Returns:
            Augmented QA pairs with spatial questions
        """
        augmented = qa_pairs.copy()
        
        # Add quadrant-based questions
        spatial_templates = [
            "How many defects are in the top-left quadrant?",
            "What defects are in the bottom-right area?",
            "Count defects in the center region.",
            "Which defects are near the edges?"
        ]
        
        # This would require spatial analysis of detections
        # Placeholder for now
        
        return augmented
