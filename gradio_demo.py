import gradio as gr
import torch
import numpy as np
import sys
import os
import torch.nn.functional as F

# --- CRITICAL FIX: Ensure Project Root is in the search path ---
# This guarantees Gradio can find your 'models/src' files.
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT) 
# ---------------------------------------------------------------

# Import ML model and utilities (assuming imports from train.py structure)
try:
    from models.src.models import MultimodalFusionModel
    from models.src.utils import load_model, get_device, TARGET_CLASSES
    from config import (
        MODEL_SAVE_PATH, TEXT_FEAT_DIM, AUDIO_FEAT_DIM, VIDEO_FEAT_DIM, CONTEXT_WINDOW
    )
except ImportError as e:
    print(f"FATAL ERROR: Could not load ML modules. Check imports. Error: {e}")
    sys.exit(1)


# --- GLOBAL MODEL LOADING (Caches the model for speed) ---
MODEL_PATH = "./models/final_model.pt"
FUSION_DIM = 64
DEVICE = get_device()

MODEL_CACHE = {} 

def load_global_model():
    """Initializes and loads the saved model weights once."""
    if 'model' in MODEL_CACHE:
        return MODEL_CACHE['model']
        
    try:
        model = MultimodalFusionModel(fusion_dim=FUSION_DIM).to(DEVICE)
        model = load_model(model, MODEL_PATH, DEVICE) 
        model.eval()
        MODEL_CACHE['model'] = model
        return model
    except Exception as e:
        print(f"ERROR: Model load failed. Details: {e}")
        return None

# --- PREDICTION FUNCTION (Gradio compatible) ---

def predict_emotion(input_text):
    """
    Runs model inference on input text (using simulated audio/video features).
    """
    model = load_global_model()
    if model is None:
        return "ERROR: Model not loaded. Check terminal for details.", "0%"

    # 1. Simulate Multimodal Input (Using zeros as placeholders)
    # We define all 6 features, but only pass the 3 non-contextual ones to the model call.
    text_x = torch.zeros(1, TEXT_FEAT_DIM, dtype=torch.float32).to(DEVICE)
    audio_x = torch.zeros(1, AUDIO_FEAT_DIM, dtype=torch.float32).to(DEVICE)
    video_x = torch.zeros(1, VIDEO_FEAT_DIM, dtype=torch.float32).to(DEVICE)
    # Context features are still defined but NOT used in the model call below.
    ctx_text = torch.zeros(1, CONTEXT_WINDOW, TEXT_FEAT_DIM, dtype=torch.float32).to(DEVICE)
    ctx_audio = torch.zeros(1, CONTEXT_WINDOW, AUDIO_FEAT_DIM, dtype=torch.float32).to(DEVICE)
    ctx_video = torch.zeros(1, CONTEXT_WINDOW, VIDEO_FEAT_DIM, dtype=torch.float32).to(DEVICE)

    # 2. Run Model Inference
    with torch.no_grad():
        # --- CRITICAL FIX: ONLY PASSES 3 CURRENT INPUTS ---
        # This resolves the "takes 4 positional arguments but 7 were given" error.
        emotion_logits, _ = model(text_x, audio_x, video_x) # <--- CONTEXT ARGUMENTS REMOVED HERE
        # ----------------------------------------------------

    # 3. Process Result
    emotion_prob = torch.softmax(emotion_logits, dim=1).squeeze().cpu().numpy()
    pred_index = np.argmax(emotion_prob)
    confidence = emotion_prob[pred_index] * 100
    
    return TARGET_CLASSES[pred_index].upper(), f"{confidence:.2f}%"


# --- GRADIO INTERFACE ---

iface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Textbox(lines=3, placeholder="Enter a sentence for emotion analysis...", label="Text Input"),
    outputs=[
        gr.Textbox(label="Predicted Emotion", type="text"),
        gr.Textbox(label="Confidence Level", type="text")
    ],
    title="Multimodal Emotion Detection (Gradio)",
    description="Deploys the trained PyTorch Multimodal Model. A/V features are simulated for simplicity.",
    theme="soft",
    allow_flagging="never"
)

# Launch the app
if __name__ == '__main__':
    print("Gradio App is starting...")
    load_global_model() # Pre-load the model before launching interface

    # --- FINAL DEPLOYMENT COMMAND ---
    iface.launch(inbrowser=True, share=True) 
    # --------------------------------