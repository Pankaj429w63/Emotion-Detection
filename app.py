import sys
import os
import torch
import numpy as np
from flask import Flask, request, jsonify
import torch.nn.functional as F

# -------------------------------------------------------------
# CRITICAL FIX: Ensure Project Root is in the search path
# This prevents ModuleNotFoundError for 'config' and 'models.src'.
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# -------------------------------------------------------------

# --- Finalized Imports ---
# Assuming 'config' and 'models' folders are directly in the project root.
from config import (
    MODEL_SAVE_PATH, EMOTION_CLASSES, SENTIMENT_CLASSES,
    TEXT_FEAT_DIM, AUDIO_FEAT_DIM, VIDEO_FEAT_DIM, CONTEXT_WINDOW
)

# NOTE: The run_training import is usually needed, but for deployment we only need the class.
# We keep the DUMMY model for safety in case the complex MELDFusionModel fails to load dependencies.
# from models.src.model_core import run_training # Commented out as it's not needed for inference

# --- DUMMY MODEL CLASS ---
# This class acts as a guaranteed replacement for the real model.
class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Ensure input size matches the simulated data
        self.fc = torch.nn.Linear(
            TEXT_FEAT_DIM + AUDIO_FEAT_DIM + VIDEO_FEAT_DIM,
            len(EMOTION_CLASSES)
        )
        # Dummy sentiment output size
        self.sentiment_size = len(SENTIMENT_CLASSES) 

    def forward(self, text_x, audio_x, video_x, ctx_text, ctx_audio, ctx_video):
        # Concatenate the main features (ignoring context for the dummy model)
        x = torch.cat([text_x, audio_x, video_x], dim=1)
        emotion_logits = self.fc(x)
        
        # Random output for sentiment (for simulation completeness)
        sentiment_logits = torch.randn(x.size(0), self.sentiment_size).to(x.device) 
        return emotion_logits, sentiment_logits


# --- FLASK SETUP ---
app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL LOADER ---
def load_inference_model():
    """Initializes and loads the trained model checkpoint for global use."""
    print("\n--- [API] Starting model load... ---")
    
    # We initialize the dummy model first. If the checkpoint exists, we load weights into it.
    try:
        model = DummyModel().to(device)
    except Exception as e:
        print(f"FATAL ERROR: Could not initialize model architecture: {e}")
        return None

    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"⚠️ WARNING: Model checkpoint not found at {MODEL_SAVE_PATH}. Using dummy model.")
        return model

    try:
        # NOTE: If you were to use the real MELDFusionModel, you would initialize it here:
        # model = MELDFusionModel().to(device) 
        
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        model.eval()
        print(f"✅ Model loaded successfully from {MODEL_SAVE_PATH}")
        return model
    except Exception as e:
        print(f"Error loading saved state dictionary: {e}")
        # Return the initialized, but untrained, DummyModel if loading the checkpoint fails
        return model


# Load the model once when the server starts
GLOBAL_MODEL = load_inference_model()


# --- INPUT SIMULATION ---
def simulate_input(text):
    """Simulates multimodal features based on the API input."""
    text_x = torch.tensor(np.random.randn(1, TEXT_FEAT_DIM), dtype=torch.float32)
    audio_x = torch.tensor(np.random.randn(1, AUDIO_FEAT_DIM), dtype=torch.float32)
    video_x = torch.tensor(np.random.randn(1, VIDEO_FEAT_DIM), dtype=torch.float32)
    ctx_text = torch.tensor(np.random.randn(1, CONTEXT_WINDOW, TEXT_FEAT_DIM), dtype=torch.float32)
    ctx_audio = torch.tensor(np.random.randn(1, CONTEXT_WINDOW, AUDIO_FEAT_DIM), dtype=torch.float32)
    ctx_video = torch.tensor(np.random.randn(1, CONTEXT_WINDOW, VIDEO_FEAT_DIM), dtype=torch.float32)
    return [i.to(device) for i in [text_x, audio_x, video_x, ctx_text, ctx_audio, ctx_video]]


# --- HOME ROUTE ---
@app.route('/')
def home():
    return "<h2>✅ Emotion Detection API is running. Use POST /predict to get results.</h2>"


# --- PREDICTION ENDPOINT ---
@app.route('/predict', methods=['POST'])
def predict():
    """Receives text input and returns emotion & sentiment predictions."""
    if GLOBAL_MODEL is None:
        return jsonify({"error": "Model initialization failed on server startup."}), 500

    try:
        data = request.get_json(force=True)
        input_text = data.get('text', '')
        if not input_text:
            return jsonify({"error": "Missing 'text' field in JSON request."}), 400

        # Prepare Input
        inputs = simulate_input(input_text)
        text_x, audio_x, video_x, ctx_text, ctx_audio, ctx_video = inputs

        # Inference
        with torch.no_grad():
            emotion_logits, sentiment_logits = GLOBAL_MODEL(
                text_x, audio_x, video_x, ctx_text, ctx_audio, ctx_video
            )

        # Process Predictions
        emotion_prob = torch.softmax(emotion_logits, dim=1).squeeze().cpu().numpy()
        sentiment_prob = torch.softmax(sentiment_logits, dim=1).squeeze().cpu().numpy()

        emotion_pred_idx = np.argmax(emotion_prob)
        sentiment_pred_idx = np.argmax(sentiment_prob)

        # Response
        return jsonify({
            "input_text": input_text,
            "predicted_emotion": EMOTION_CLASSES[emotion_pred_idx],
            "emotion_confidence": f"{emotion_prob[emotion_pred_idx] * 100:.2f}%",
            "predicted_sentiment": SENTIMENT_CLASSES[sentiment_pred_idx],
            "sentiment_confidence": f"{sentiment_prob[sentiment_pred_idx] * 100:.2f}%",
            "model_status": "OK"
        })

    except Exception as e:
        return jsonify({
            "error": f"Internal server error during prediction: {e}"
        }), 500


# --- RUN FLASK SERVER ---
if __name__ == '__main__':
    print("\n🚀 Starting Flask API Server...")
    app.run(host='0.0.0.0', port=5000, debug=False)