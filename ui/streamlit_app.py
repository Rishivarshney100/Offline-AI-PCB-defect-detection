"""
Streamlit-based UI for PCB Defect AI Agent
"""

import streamlit as st
import sys
from pathlib import Path
import tempfile
import os
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ui.app import create_agent
from agent.agent import PCBDefectAgent
from src.utils.visualization import draw_bounding_boxes


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="PCB Defect AI Agent",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 PCB Defect AI Agent")
    st.markdown("Upload a PCB image and ask questions about defects in natural language")
    
    # Initialize agent (cached)
    @st.cache_resource
    def load_agent():
        model_path = st.session_state.get("model_path", "models/weights/best.pt")
        return create_agent(model_path=model_path)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        model_path = st.text_input(
            "Model Path",
            value="models/weights/best.pt",
            help="Path to trained YOLOv5 model"
        )
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05
        )
        
        if st.button("Reload Agent"):
            st.cache_resource.clear()
            st.session_state["model_path"] = model_path
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Image Upload")
        uploaded_file = st.file_uploader(
            "Choose a PCB image",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Upload an image of a PCB to analyze"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            # Store image path in session state
            st.session_state["current_image"] = tmp_path
            
            # Show original image only (no auto-detection) - use same display method as visualization
            image = cv2.imread(tmp_path)
            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Resize to max width for consistent display size
                max_width = 600
                height, width = image_rgb.shape[:2]
                if width > max_width:
                    scale = max_width / width
                    new_width = max_width
                    new_height = int(height * scale)
                    image_rgb = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
                # Use same display settings as visualization for consistent sizing
                st.image(image_rgb, caption="Uploaded PCB Image", use_container_width=False, channels="RGB")
    
    with col2:
        st.header("Query")
        # Pre-fill query if example was clicked
        default_query = st.session_state.get("query", "")
        query = st.text_area(
            "Ask about defects in natural language",
            value=default_query,
            placeholder="e.g., 'How many defects are there?', 'What types of defects?', 'Where are the high severity defects?'",
            height=100
        )
        # Clear the session state query after using it
        if "query" in st.session_state:
            del st.session_state["query"]
        
        if st.button("Analyze", type="primary"):
            if "current_image" not in st.session_state:
                st.error("Please upload an image first!")
            elif not query.strip():
                st.error("Please enter a query!")
            else:
                try:
                    agent = load_agent()
                    with st.spinner("Analyzing image and processing query..."):
                        # Get detection results
                        results = agent.analyze_image(st.session_state["current_image"])
                        defects = results.get("defects", [])
                        
                        # Generate response
                        response = agent.answer_query(
                            image_path=st.session_state["current_image"],
                            query=query
                        )
                    
                    st.success("Analysis Complete!")
                    
                    # Display visualization and response side by side
                    viz_col, response_col = st.columns([1.2, 0.8])
                    
                    with viz_col:
                        st.markdown("### 🔍 Visualized Results")
                        # Load image and draw bounding boxes
                        image = cv2.imread(st.session_state["current_image"])
                        if image is not None:
                            # Convert BGR to RGB for display
                            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            
                            # Draw bounding boxes with confidence and severity
                            if defects:
                                annotated_image = draw_bounding_boxes(
                                    image_rgb,
                                    defects,
                                    show_confidence=True,
                                    show_severity=True
                                )
                                # Resize to same max width as uploaded image for consistent sizing
                                max_width = 600
                                height, width = annotated_image.shape[:2]
                                if width > max_width:
                                    scale = max_width / width
                                    new_width = max_width
                                    new_height = int(height * scale)
                                    annotated_image = cv2.resize(annotated_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                                # Use same display settings as uploaded image for consistent sizing
                                st.image(annotated_image, caption="Detected Defects", use_container_width=False, channels="RGB")
                                
                                # Show defect count summary
                                defect_count = len(defects)
                                defect_types = {}
                                for defect in defects:
                                    defect_type = defect.get('defect_type', 'Unknown')
                                    defect_types[defect_type] = defect_types.get(defect_type, 0) + 1
                                
                                st.markdown(f"**Total Defects Found: {defect_count}**")
                                if defect_types:
                                    st.markdown("**By Type:**")
                                    for dtype, count in defect_types.items():
                                        st.markdown(f"- {dtype}: {count}")
                            else:
                                # Resize to same max width as uploaded image for consistent sizing
                                max_width = 600
                                height, width = image_rgb.shape[:2]
                                if width > max_width:
                                    scale = max_width / width
                                    new_width = max_width
                                    new_height = int(height * scale)
                                    image_rgb = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
                                st.image(image_rgb, caption="No Defects Detected", use_container_width=False, channels="RGB")
                                st.info("No defects were detected in this image.")
                    
                    with response_col:
                        st.markdown("### 💬 AI Response")
                        st.info(response)
                        
                        # Show detection results in expandable section
                        with st.expander("📊 View Detailed JSON Results"):
                            st.json(results)
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
    
    # Example queries
    st.markdown("---")
    st.header("Example Queries")
    example_queries = [
        "How many defects are there?",
        "What types of defects were found?",
        "Where are the defects located?",
        "What is the severity of the defects?",
        "Are there any missing hole defects?",
        "Tell me about all the defects",
        "What is the confidence of the detections?"
    ]
    
    cols = st.columns(3)
    for i, example in enumerate(example_queries):
        with cols[i % 3]:
            if st.button(example, key=f"example_{i}"):
                if "current_image" in st.session_state:
                    st.session_state["query"] = example
                    st.rerun()


if __name__ == "__main__":
    main()
