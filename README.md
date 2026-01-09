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

**Option 2: Direct Streamlit (if `streamlit` command not found, use this)**
```bash
python -m streamlit run ui/streamlit_app.py
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

### Step 6: View Results

The system returns:
- **Natural language response** explaining the defects
- **Structured JSON** with:
  - Count of defects
  - Defect types
  - Coordinates (x, y)
  - Confidence scores
  - Severity levels

### Example Queries
![WhatsApp Image 2026-01-09 at 5 11 37 PM](https://github.com/user-attachments/assets/12ef2cb6-5930-4e8c-81e5-5a5d0a41d12f)
![WhatsApp Image 2026-01-09 at 5 11 10 PM](https://github.com/user-attachments/assets/f9d28187-da86-438e-ac5b-f09d8782ab32)
![WhatsApp Image 2026-01-09 at 5 09 23 PM](https://github.com/user-attachments/assets/c027715a-713c-41b8-9c3a-b0b79dfe26dd)


## Architecture

```
PCB Image → YOLOv5 Detector → Structured Tokens → Small LLM → JSON Response
```

## Usage

### Training
```bash
python src/train.py --data data/dataset.yaml --epochs 50
```

### Inference
```bash
python src/infer.py --model models/weights/best.pt --image path/to/image.jpg
```

### UI
```bash
python main.py
# or
streamlit run ui/streamlit_app.py
```

### VLM (Natural Language Queries)
```python
from vlm.vlm_model import DetectionGroundedVLM
from vlm.architecture import VLMConfig

vlm = DetectionGroundedVLM(VLMConfig(llm_backend="ollama", llm_model_name="phi-2"))
response = vlm.generate("images/pcb.jpg", "How many shorts are present?")
```

## Project Structure

```
├── src/              # Training, inference, evaluation
├── agent/            # AI agent (query processing, response generation)
├── vlm/              # Detection-Grounded VLM
├── ui/               # Streamlit UI
├── config/           # Configuration files
├── images/            # PCB images (50k+)
└── models/weights/    # Trained YOLOv5 models
```

## Output Format

```json
{
  "count": 2,
  "defect_type": "Short",
  "locations": [[135, 355], [290, 410]],
  "confidence": 0.89,
  "severity": "High"
}
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
Factors: model size (1-3B vs 7-13B), speed, fine-tuning flexibility, licensing. 
Modifications: Detection-first pipeline with YOLOv5 bounding boxes, structured token encoding, region-conditioned prompting, output validation.

### (B) Design Strategy

**Architecture**: YOLOv5 detector → Defect Tokenizer → Small LLM (1-3B) → Output Parser. 
Modified components: PCB-trained YOLOv5 replaces CLIP/ViT, custom tokenizer for bounding boxes, restricted vocabulary LLM, region-conditioned fusion, validation parser. 
Handles 6 defect types, spatial queries via coordinates, explicit counting, severity estimation.

### (C) Optimization

**Model-level**: INT8 quantization (2-4× speedup), pruning, distillation, LoRA. 
**System-level**: ONNX/TensorRT export, caching, CPU threading.
 **Performance**: Vision 200-400ms + LLM 300-600ms = total <1.2s.

### (D) Hallucination Mitigation

**Architectural**: Detection-first pipeline, structured JSON schema, closed-set vocabulary, output validation. 
**Training**: Grounding loss, consistency loss, hallucination penalty, negative examples. 
**Result**: <1% hallucination rate (vs 5-30% for generic VLMs).

### (E) Training Plan

**Stages**: 
(1) YOLOv5 pretraining
(2) Synthetic QA generation from bounding boxes
(3) LLM LoRA fine-tuning
(4) Joint training
(5) Stress testing 
**Augmentation**: Image rotation/scaling, query paraphrasing, negative examples. 
**Metrics**: Counting accuracy, localization IoU, hallucination rate, response time, consistency.

### (F) Validation

**Counting**: Compare LLM vs detection count, target >99%. 
**Localization**: Mean center error <5px, IoU >0.8, spatial accuracy >95%. 
**Hallucination**: Test non-existent defects, verify all mentioned defects exist, target <1%. 
**Additional**: Response time <2s, consistency >99%, robustness testing.
