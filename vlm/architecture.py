from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DefectDetection:
    defect_type: str
    bbox: Tuple[float, float, float, float]
    center: Tuple[float, float]
    confidence: float
    severity: str


@dataclass
class VLMConfig:
    detector_model_path: str = "models/weights/best.pt"
    detector_conf_threshold: float = 0.25
    llm_backend: str = "ollama"
    llm_model_name: str = "phi-2"
    llm_base_url: Optional[str] = None
    llm_temperature: float = 0.1
    llm_max_tokens: int = 256
    use_quantization: bool = True
    quantization_type: str = "int8"
    use_onnx: bool = True
    output_format: str = "json"
    validate_output: bool = True


class DetectionGroundedVLM:
    """
    Custom Detection-Grounded Vision-Language Model for PCB Inspection.
    
    Architecture:
        PCB Image → YOLOv5 Detector → Structured Defect Tokens → Small LLM → JSON Response
    
    Key Features:
        - Detection-first pipeline prevents hallucination
        - Explicit bounding box grounding
        - Structured JSON output
        - Fast inference (<2s on CPU)
    """
    
    def __init__(self, config: VLMConfig):
        """
        Initialize the Detection-Grounded VLM.
        
        Args:
            config: VLM configuration
        """
        self.config = config
        
        # Components will be initialized lazily
        self.detector = None
        self.defect_tokenizer = None
        self.llm = None
        self.output_parser = None
        
    def _initialize_components(self):
        """Lazy initialization of components."""
        if self.detector is None:
            from src.infer import PCBDefectDetector
            self.detector = PCBDefectDetector(
                model_path=self.config.detector_model_path,
                conf_threshold=self.config.detector_conf_threshold
            )
        
        if self.defect_tokenizer is None:
            from vlm.tokenizer import DefectTokenizer
            self.defect_tokenizer = DefectTokenizer()
        
        if self.llm is None:
            from vlm.llm_wrapper import LLMWrapper
            self.llm = LLMWrapper(
                backend=self.config.llm_backend,
                model_name=self.config.llm_model_name,
                base_url=self.config.llm_base_url,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens
            )
        
        if self.output_parser is None:
            from vlm.output_parser import OutputParser
            self.output_parser = OutputParser(validate=self.config.validate_output)
    
    def encode_detections(self, detections: List[Dict]) -> str:
        """
        Convert detection results to structured tokens for LLM.
        
        Args:
            detections: List of detection dictionaries from YOLOv5
            
        Returns:
            Structured prompt string
        """
        self._initialize_components()
        return self.defect_tokenizer.encode(detections)
    
    def generate_response(self, image_path: str, query: str) -> Dict:
        """
        Generate structured response to natural language query.
        
        Args:
            image_path: Path to PCB image
            query: Natural language query about defects
            
        Returns:
            Dictionary with structured response (JSON format)
        """
        self._initialize_components()
        
        # Step 1: Detect defects in image
        detection_results = self.detector.predict(image_path)
        defects = detection_results.get("defects", [])
        
        # Step 2: Encode detections as structured tokens
        detection_prompt = self.encode_detections(defects)
        
        # Step 3: Create full prompt for LLM
        full_prompt = self._create_prompt(detection_prompt, query)
        
        # Step 4: Generate response from LLM
        llm_response = self.llm.generate(full_prompt)
        
        # Step 5: Parse and validate output
        parsed_output = self.output_parser.parse(
            llm_response, 
            query=query,
            detections=defects
        )
        
        return parsed_output
    
    def _create_prompt(self, detection_prompt: str, query: str) -> str:
        """
        Create full prompt for LLM with detection context and query.
        
        Args:
            detection_prompt: Structured detection information
            query: User's natural language query
            
        Returns:
            Complete prompt string
        """
        prompt_template = """You are a PCB defect inspection assistant. Answer questions about detected defects using ONLY the information provided.

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
        
        return prompt_template.format(
            detections=detection_prompt,
            query=query
        )
    
    def generate_natural_language_response(self, image_path: str, query: str) -> str:
        """
        Generate natural language response (backward compatibility).
        
        Args:
            image_path: Path to PCB image
            query: Natural language query
            
        Returns:
            Natural language response string
        """
        structured_response = self.generate_response(image_path, query)
        
        # Convert structured JSON to natural language
        return self._json_to_natural_language(structured_response, query)
    
    def _json_to_natural_language(self, response: Dict, query: str) -> str:
        """
        Convert structured JSON response to natural language.
        
        Args:
            response: Structured JSON response
            query: Original query
            
        Returns:
            Natural language string
        """
        count = response.get("count", 0)
        defect_type = response.get("defect_type")
        locations = response.get("locations", [])
        confidence = response.get("confidence", 0.0)
        
        if count == 0:
            return "No matching defects were found in this PCB image."
        
        # Build natural language response
        parts = []
        
        if defect_type:
            parts.append(f"I found {count} {defect_type.lower()} defect(s)")
        else:
            parts.append(f"I found {count} defect(s)")
        
        if locations:
            if len(locations) == 1:
                x, y = locations[0]
                parts.append(f"located at coordinates ({x:.1f}, {y:.1f})")
            else:
                parts.append(f"at {len(locations)} locations")
        
        if confidence > 0:
            parts.append(f"with confidence {confidence:.2%}")
        
        return ". ".join(parts) + "."


# Architecture Flow Diagram (for documentation)
ARCHITECTURE_FLOW = """
┌─────────────────┐
│   PCB Image     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  YOLOv5 Detector│  ← Vision Encoder (Domain-Specific)
│  (PCB-trained)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Bounding Boxes │  ← Explicit Detection Results
│  + Features     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Defect Tokenizer│  ← Convert to Structured Tokens
│  (Structured    │
│   Prompt)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Small LLM     │  ← Language Model (Phi-2/Qwen-1.5-1.8B)
│  (1-3B params)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Output Parser  │  ← JSON Validation + Grounding Check
│  (Validation)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  JSON Response  │  ← Structured Output
│  (Validated)    │
└─────────────────┘
"""
