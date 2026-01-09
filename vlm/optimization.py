"""
Optimization utilities for VLM inference speed
Includes quantization, ONNX export, and inference optimization
"""

from typing import Optional, Dict, Any
import os
from pathlib import Path


class ModelOptimizer:
    """
    Optimizes models for faster inference.
    Supports quantization, ONNX export, and other optimizations.
    """
    
    def __init__(self):
        """Initialize model optimizer."""
        pass
    
    def quantize_llm(self, model_path: str, output_path: str, 
                    quantization_type: str = "int8") -> str:
        """
        Quantize LLM model for faster inference.
        
        Args:
            model_path: Path to original model
            output_path: Path to save quantized model
            quantization_type: "int8" or "int4"
        
        Returns:
            Path to quantized model
        """
        # This is a placeholder - actual quantization depends on backend
        # For Ollama: models are already optimized
        # For Transformers: use bitsandbytes or onnxruntime
        
        if quantization_type == "int8":
            return self._quantize_int8(model_path, output_path)
        elif quantization_type == "int4":
            return self._quantize_int4(model_path, output_path)
        else:
            raise ValueError(f"Unknown quantization type: {quantization_type}")
    
    def _quantize_int8(self, model_path: str, output_path: str) -> str:
        """Quantize to INT8."""
        # Placeholder - actual implementation depends on model format
        print(f"Quantizing {model_path} to INT8...")
        print("Note: For Ollama, models are pre-optimized.")
        print("For Transformers, use bitsandbytes or onnxruntime.")
        return output_path
    
    def _quantize_int4(self, model_path: str, output_path: str) -> str:
        """Quantize to INT4."""
        print(f"Quantizing {model_path} to INT4...")
        print("Note: INT4 quantization requires specific model support.")
        return output_path
    
    def export_detector_onnx(self, model_path: str, output_path: str, 
                            img_size: int = 640) -> str:
        """
        Export YOLOv5 detector to ONNX format for faster inference.
        
        Args:
            model_path: Path to PyTorch model (.pt)
            output_path: Path to save ONNX model (.onnx)
            img_size: Input image size
        
        Returns:
            Path to ONNX model
        """
        try:
            from ultralytics import YOLO
            
            print(f"Exporting {model_path} to ONNX...")
            model = YOLO(model_path)
            model.export(format="onnx", imgsz=img_size, simplify=True)
            
            # Find the exported file
            base_name = Path(model_path).stem
            onnx_file = Path(model_path).parent / f"{base_name}.onnx"
            
            if onnx_file.exists():
                # Move to output path
                import shutil
                shutil.move(str(onnx_file), output_path)
                print(f"ONNX model saved to {output_path}")
                return output_path
            else:
                raise FileNotFoundError(f"ONNX export failed: {onnx_file} not found")
        
        except ImportError:
            print("Warning: ultralytics not available for ONNX export")
            return model_path
        except Exception as e:
            print(f"Warning: ONNX export failed: {e}")
            return model_path
    
    def optimize_inference(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply inference optimizations.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            Optimized configuration
        """
        optimized = config.copy()
        
        # Enable ONNX if available
        if config.get("use_onnx", True):
            detector_path = config.get("detector_model_path", "models/weights/best.pt")
            onnx_path = detector_path.replace(".pt", ".onnx")
            
            if not os.path.exists(onnx_path):
                try:
                    self.export_detector_onnx(detector_path, onnx_path)
                    optimized["detector_model_path"] = onnx_path
                except Exception as e:
                    print(f"ONNX export failed, using original model: {e}")
        
        # Set quantization
        if config.get("use_quantization", True):
            optimized["quantization"] = config.get("quantization_type", "int8")
        
        return optimized


class InferenceOptimizer:
    """
    Optimizes inference pipeline for speed.
    Includes caching, batching, and other techniques.
    """
    
    def __init__(self):
        """Initialize inference optimizer."""
        self.prompt_cache = {}
        self.detection_cache = {}
    
    def cache_prompt(self, detections: list, query: str, prompt: str):
        """
        Cache generated prompt for reuse.
        
        Args:
            detections: Detection results
            query: User query
            prompt: Generated prompt
        """
        # Create cache key from detections hash and query
        cache_key = self._create_cache_key(detections, query)
        self.prompt_cache[cache_key] = prompt
    
    def get_cached_prompt(self, detections: list, query: str) -> Optional[str]:
        """
        Get cached prompt if available.
        
        Args:
            detections: Detection results
            query: User query
        
        Returns:
            Cached prompt or None
        """
        cache_key = self._create_cache_key(detections, query)
        return self.prompt_cache.get(cache_key)
    
    def _create_cache_key(self, detections: list, query: str) -> str:
        """Create cache key from detections and query."""
        import hashlib
        
        # Create hash from detection summary and query
        det_summary = str(len(detections)) + str(sorted([d.get("defect_type") for d in detections]))
        key_str = det_summary + query.lower()
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def optimize_batch(self, images: list, queries: list) -> list:
        """
        Optimize batch processing.
        
        Args:
            images: List of image paths
            queries: List of queries
        
        Returns:
            List of results
        """
        # Group by image for detection caching
        image_groups = {}
        for i, img_path in enumerate(images):
            if img_path not in image_groups:
                image_groups[img_path] = []
            image_groups[img_path].append(i)
        
        # Process detections once per image
        detection_results = {}
        for img_path in image_groups.keys():
            # This would use the detector - placeholder
            detection_results[img_path] = None
        
        # Process queries with cached detections
        results = []
        for i, query in enumerate(queries):
            img_path = images[i]
            # Use cached detection
            # Process query
            results.append(None)  # Placeholder
        
        return results
