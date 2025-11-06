import torch
import torch.nn.functional as F
import random
import sys
import os # <-- Needed for path manipulation

# --- CRITICAL FIX: Add Project Root to Path ---
# This line calculates the path to the 'D:\Emotion Dataset' directory (the parent of 'models')
# and adds it to the list of folders Python searches for modules.
# It resolves the 'ModuleNotFoundError' when running the script directly.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
# ---------------------------------------------

# Import necessary components from your project structure
# NOTE: These imports now work because the project root is in sys.path
from models.src.models import MultimodalFusionModel 
from models.src.utils import load_model, get_device, TARGET_CLASSES

# --- Configuration ---
MODEL_PATH = "./models/final_model.pt"
FUSION_DIM = 64 # Must match the fusion_dim used in models.py (which was 64)

# --- Simulation Functions ---
# These functions create dummy data that matches the size your model expects.

def preprocess_text(sentence, device):
    """Placeholder: Creates dummy input tensors for BERT (size 768)."""
    # NOTE: The size 768 is the standard BERT output dimension.
    print(f"  > Simulating text processing for: '{sentence}'")
    
    # Creates a dummy tensor of the correct shape (Batch=1, Dim=768 is assumed output of BERT)
    dummy_text_input = {
        'input_ids': torch.zeros(1, 128, dtype=torch.long).to(device),
        'attention_mask': torch.zeros(1, 128, dtype=torch.long).to(device)
    }
    return dummy_text_input

def preprocess_audio_video(device):
    """Placeholder: Creates dummy sequential features for Audio and Video LSTMs."""
    # Sizes based on the initialized dimensions in your models.py
    print("  > Using dummy audio/video features (simulation).")
    
    # randn creates random numbers (Batch=1, Sequence_Length=1, Feature_Dim=X)
    dummy_audio = torch.randn(1, 1, 128).to(device) 
    dummy_video = torch.randn(1, 1, 512).to(device) 
    return dummy_audio, dummy_video


def predict_emotion(sentence):
    """
    Loads the trained model, prepares dummy data, and predicts the emotion.
    """
    device = get_device()
    
    # 1. Initialize the Model Structure
    print("\n[STEP 1: Initializing and Loading Model]")
    model = MultimodalFusionModel(fusion_dim=FUSION_DIM).to(device)
    
    # 2. Load the SAVED WEIGHTS (The 'Brain')
    try:
        # Assuming MODEL_PATH is relative to the project root
        model = load_model(model, MODEL_PATH, device)
    except Exception as e:
        print(f"FATAL ERROR loading model: {e}")
        print("Ensure final_model.pt exists in the ./models folder.")
        return

    model.eval() # Set model to evaluation mode
    
    # 3. Preprocess Input Data
    print("\n[STEP 2: Preparing Input Data (Multimodal)]")
    text_input = preprocess_text(sentence, device)
    audio_x, video_x = preprocess_audio_video(device)
    
    # 4. Get Prediction
    print("\n[STEP 3: Generating Prediction]")
    with torch.no_grad():
        # Pass all three modalities (text, audio, video) to the model
        logits = model(text_input, audio_x, video_x)
        
        # Convert logits (raw output) to probabilities using Softmax
        probabilities = F.softmax(logits, dim=1)
        
        # Get the index and confidence of the most likely emotion
        pred_index = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][pred_index].item() * 100
        predicted_emotion = TARGET_CLASSES[pred_index]
    
    # 5. Output Results
    print(f"\n" + "="*40)
    print(f"| INPUT SENTENCE: '{sentence}'")
    print(f"| PREDICTED EMOTION: {predicted_emotion.upper()}")
    print(f"| CONFIDENCE: {confidence:.2f}%")
    print("="*40)


if __name__ == "__main__":
    
    # Example 1: An utterance the model is typically good at (high support class)
    predict_emotion("Oh my god, I haven't seen you in ages, this is amazing!")
    
    print("\n" + "*"*50 + "\n")
    
    # Example 2: An utterance that is ambiguous (more challenging)
    predict_emotion("Well, that's just perfect, isn't it?")