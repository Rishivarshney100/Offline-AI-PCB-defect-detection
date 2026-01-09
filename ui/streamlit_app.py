"""
Streamlit-based UI for PCB Defect AI Agent
"""

import streamlit as st
import sys
from pathlib import Path
import tempfile
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from ui.app import create_agent
from agent.agent import PCBDefectAgent


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
            
            st.image(uploaded_file, caption="Uploaded PCB Image", use_container_width=True)
            
            # Store image path in session state
            st.session_state["current_image"] = tmp_path
    
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
                        response = agent.answer_query(
                            image_path=st.session_state["current_image"],
                            query=query
                        )
                    
                    st.success("Analysis Complete!")
                    st.markdown("### Response:")
                    st.info(response)
                    
                    # Show detection results
                    with st.expander("View Detailed Detection Results"):
                        results = agent.analyze_image(st.session_state["current_image"])
                        st.json(results)
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
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
