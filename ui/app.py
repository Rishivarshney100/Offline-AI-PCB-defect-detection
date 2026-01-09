"""
Main UI Application for PCB Defect AI Agent
Supports multiple UI frameworks (Streamlit, Gradio, Flask)
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.agent import PCBDefectAgent


class UIFramework:
    """Enum-like class for UI framework options"""
    STREAMLIT = "streamlit"
    GRADIO = "gradio"
    FLASK = "flask"


def create_agent(model_path: str = "models/weights/best.pt", 
                conf_threshold: float = 0.25,
                device: str = "",
                llm_config: dict = None) -> PCBDefectAgent:
    """
    Create and initialize the PCB Defect Agent.
    
    Args:
        model_path: Path to trained model
        conf_threshold: Confidence threshold
        device: Device for inference
        llm_config: LLM configuration
        
    Returns:
        Initialized PCBDefectAgent instance
    """
    return PCBDefectAgent(
        model_path=model_path,
        conf_threshold=conf_threshold,
        device=device,
        llm_config=llm_config
    )


# UI implementations will be in separate files
# ui/streamlit_app.py
# ui/gradio_app.py
# ui/flask_app.py
