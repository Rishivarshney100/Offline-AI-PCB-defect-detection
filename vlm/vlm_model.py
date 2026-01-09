from typing import Dict, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vlm.architecture import DetectionGroundedVLM, VLMConfig
from vlm.tokenizer import DefectTokenizer
from vlm.llm_wrapper import LLMWrapper
from vlm.output_parser import OutputParser


class DetectionGroundedVLM:
    def __init__(self, config: Optional[VLMConfig] = None, **kwargs):
        self.config = config or VLMConfig(**kwargs)
        self._init_detector()
        self._init_tokenizer()
        self._init_llm()
        self._init_parser()
    
    def _init_detector(self):
        from src.infer import PCBDefectDetector
        self.detector = PCBDefectDetector(
            model_path=self.config.detector_model_path,
            conf_threshold=self.config.detector_conf_threshold
        )
    
    def _init_tokenizer(self):
        self.tokenizer = DefectTokenizer()
    
    def _init_llm(self):
        self.llm = LLMWrapper(
            backend=self.config.llm_backend,
            model_name=self.config.llm_model_name,
            base_url=self.config.llm_base_url,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens
        )
    
    def _init_parser(self):
        self.parser = OutputParser(validate=self.config.validate_output)
    
    def generate(self, image_path: str, query: str) -> Dict:
        detection_results = self.detector.predict(image_path)
        defects = detection_results.get("defects", [])
        detection_prompt = self.tokenizer.encode(defects)
        full_prompt = self._create_prompt(detection_prompt, query)
        llm_response = self.llm.generate(full_prompt)
        return self.parser.parse(llm_response, query=query, detections=defects)
    
    def _create_prompt(self, detection_prompt: str, query: str) -> str:
        template = """You are a PCB defect inspection assistant. Answer questions about detected defects using ONLY the information provided.

{detections}

Question: {query}

Answer in JSON format only. Use this schema:
{{
  "count": <number>,
  "defect_type": "<type>",
  "locations": [[x1, y1], [x2, y2], ...],
  "confidence": <0.0-1.0>,
  "severity": "<Low|Medium|High>"
}}

If no defects match the query, return:
{{
  "count": 0,
  "defect_type": null,
  "locations": [],
  "confidence": 0.0,
  "severity": null
}}

JSON Answer:"""
        return template.format(detections=detection_prompt, query=query)
    
    def generate_natural_language(self, image_path: str, query: str) -> str:
        return self._json_to_natural_language(self.generate(image_path, query), query)
    
    def _json_to_natural_language(self, response: Dict, query: str) -> str:
        count = response.get("count", 0)
        defect_type = response.get("defect_type")
        locations = response.get("locations", [])
        confidence = response.get("confidence", 0.0)
        
        if count == 0:
            return "No matching defects were found in this PCB image."
        
        parts = [f"I found {count} {defect_type.lower() if defect_type else ''} defect(s)".strip()]
        if locations:
            if len(locations) == 1:
                parts.append(f"located at coordinates ({locations[0][0]:.1f}, {locations[0][1]:.1f})")
            else:
                parts.append(f"at {len(locations)} locations")
        if confidence > 0:
            parts.append(f"with confidence {confidence:.2%}")
        return ". ".join(parts) + "."
