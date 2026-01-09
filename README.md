# Offline AI for PCB Defects

Automated PCB defect detection with natural language query interface using YOLOv5 and custom Detection-Grounded VLM.

Note: This Project is based on top of the second task: https://github.com/Rishivarshney100/PCB-defects-Detection

## Features

- **Defect Detection**: 6 types (Missing Hole, Mouse Bite, Open Circuit, Short, Spur, Spurious Copper)
- **Natural Language Queries**: Ask questions about defects in plain English
- **VLM Integration**: Detection-grounded Vision-Language Model (<2s inference, <1% hallucination)
- **Web UI**: Streamlit interface
- **Offline**: No cloud dependencies

## Usage

### Step 1: Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt
```

### Step 2: Setup Ollama (for VLM)

**Windows Installation:**
1. Download Ollama from: https://ollama.ai/download
2. Run the installer (OllamaSetup.exe)
3. Restart your terminal/PowerShell
4. Verify installation:
   ```powershell
   ollama --version
   ```

**Mac/Linux Installation:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Pull LLM Model:**
```bash
ollama pull phi
```
### Training
```bash
python src/train.py --data data/dataset.yaml --epochs 50
```

### Inference
```bash
python src/infer.py --model models/weights/best.pt --image path/to/image.jpg
```

### Step 3: Launch the UI

**Option 1: Using main.py**
```bash
python main.py
```

The UI will open at `http://localhost:7860`

### Step 5: Use the Interface

1. **Upload PCB Image**: Click "Browse files" and select a PCB image
2. **Ask Questions**: Type natural language queries like:
   - "How many defects are there?"
   - "What types of defects are present?"
   - "Where are the shorts located?"
   - "How many missing holes are in this image?"
   - "What is the severity of the defects?"
3. **View Results**: See the response with defect counts, locations, and details

### Example Queries
![WhatsApp Image 2026-01-09 at 5 11 37 PM](https://github.com/user-attachments/assets/12ef2cb6-5930-4e8c-81e5-5a5d0a41d12f)
![WhatsApp Image 2026-01-09 at 5 11 10 PM](https://github.com/user-attachments/assets/f9d28187-da86-438e-ac5b-f09d8782ab32)
![WhatsApp Image 2026-01-09 at 5 09 23 PM](https://github.com/user-attachments/assets/c027715a-713c-41b8-9c3a-b0b79dfe26dd)


## Architecture

```
PCB Image → YOLOv5 Detector → Structured Tokens → Small LLM → JSON Response
```

## Project Structure

```
├── src/                      # Training, inference, evaluation (already there in the 2nd task)
│   ├── train.py             # YOLOv5 training script
│   ├── infer.py              # Inference and defect detection
│   ├── evaluate.py           # Model evaluation metrics
│   ├── data_preparation.py   # Dataset preparation utilities
│   ├── severity_estimator.py # Defect severity estimation
│   └── utils/                # Utility functions
│       ├── visualization.py # Defect visualization with bounding boxes
│       └── metrics.py        # Evaluation metrics
├── agent/                    # Offline AI agent
│   ├── agent.py              # Main PCB defect agent
│   ├── query_processor.py    # NLP processing
│   └── response_generator.py # Response generation
├── vlm/                      # Custom Detection-Grounded VLM
│   ├── vlm_model.py          # Main VLM model implementation
│   ├── architecture.py       # VLM architecture and config
│   ├── tokenizer.py          # Defect tokenizer for structured prompts
│   ├── llm_wrapper.py        # LLM backend wrapper (Ollama/Transformers)
│   ├── output_parser.py      # JSON output parser and validator
│   ├── optimization.py       # Model optimization utilities
│   ├── inference_optimizer.py # Inference speed optimization
│   ├── hallucination_control.py # Hallucination mitigation
│   ├── benchmark.py          # Performance benchmarking
│   ├── training/              # VLM training pipeline
│   │   ├── trainer.py        # Multi-stage trainer
│   │   ├── qa_generator.py   # Synthetic QA pair generation
│   │   └── augmentation.py   # Data augmentation
│   └── evaluation/            # VLM evaluation
│       ├── evaluator.py      # Evaluation metrics
│       ├── validate.py       # Validation scripts
│       └── metrics_dashboard.py # Metrics visualization
├── ui/                       # Streamlit UI
│   ├── streamlit_app.py      # Main Streamlit application
│   └── app.py                # UI helper functions
├── config/                   # Configuration files
│   └── agent_config.yaml     # Agent and VLM configuration
├── images/                    # PCB images (50k+: Could get only 13k, so used rotation script to rotate them with different angles to get 50k images)
├── models/                    # Model weights
│   └── weights/
│       └── best.pt           # Trained YOLOv5 model
├── main.py                    # Application entry point
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Requirements

- Python 3.8+
- PyTorch, Ultralytics (YOLOv5)
- Ollama (for VLM) or use rule-based mode
- See `requirements.txt` for full list

---

## Deliverables

### (A) Model Selection

**Choice: Custom Detection-Grounded VLM** over LLaVA/BLIP-2/Qwen-VL for <1% hallucination (vs 5-30%), <2s inference (vs 5-20s), and explicit bounding box localization. 
Factors: Small model size (1-3B vs 7-13B), faster speed, fine-tuning flexibility

### (B) Design Strategy

**Architecture**: YOLOv5 detector → Defect Tokenizer → Small LLM (1-3B) → Output Parser. 
Modified components: PCB-trained YOLOv5 replaces CLIP/ViT, custom tokenizer for bounding boxes, restricted vocabulary LLM, region-conditioned fusion, validation parser

### (C) Optimization

**Model-level**: INT8 quantization (2-4× speedup), pruning, distillation, LoRA
**System-level**: ONNX/TensorRT export, caching, CPU threading
 **Performance**: Vision 200-400ms + LLM 300-600ms = total <1.2s

### (D) Hallucination Mitigation

**Approach used**: Detection-first pipeline, structured JSON schema, closed-set vocabulary, output validation
**Training measures**: Grounding loss, consistency loss, hallucination penalty, negative examples
**Result**: <1% hallucination rate (vs 5-30% for generic VLMs)

### (E) Training Plan

**Stages**: 
(1) YOLOv5 pretraining
(2) Synthetic QA generation from bounding boxes
(3) LLM LoRA fine-tuning
(4) Joint training
(5) Stress testing 
**Augmentation**: Image rotation/scaling, query paraphrasing, negative examples
**Metrics**: Counting accuracy, localization IoU, hallucination rate, response time, consistency

### (F) Validation

**Counting**: Compare LLM vs detection count, target >99%
**Localization**: Mean center error <5px, IoU >0.8, spatial accuracy >95%
**Hallucination**: Test non-existent defects, verify all mentioned defects exist, target <1%
**Additional**: Response time <2s, consistency >99%, robustness testing
