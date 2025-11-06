import os
from dotenv import load_dotenv

# Load environment variables (like MODEL_DIR)
load_dotenv()

# --- FILE PATHS ---
MODEL_SAVE_PATH = os.environ.get("MODEL_DIR", "./models/") + "meld_fusion_model.pt"

# --- MELD CLASSES ---
EMOTION_CLASSES = ["Anger", "Disgust", "Fear", "Joy", "Neutral", "Sadness", "Surprise"]
SENTIMENT_CLASSES = ["Positive", "Negative", "Neutral"]
NUM_EMOTION = len(EMOTION_CLASSES)
NUM_SENTIMENT = len(SENTIMENT_CLASSES)

# --- SIMULATION & ARCHITECTURE PARAMETERS ---
CONTEXT_WINDOW = 5          # Number of previous utterances to consider for context
TEXT_FEAT_DIM = 768         # e.g., BERT embedding size
AUDIO_FEAT_DIM = 128        # e.g., Wav2Vec 2.0 or aggregate MFCC size
VIDEO_FEAT_DIM = 512        # e.g., ResNet/FaceNet feature size
CONTEXT_HIDDEN_DIM = 64     # Hidden size for context LSTMs

# --- TRAINING HYPERPARAMETERS ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 5
NUM_TOTAL_SAMPLES = 2560    # Total samples for simulated data
TRAIN_RATIO = 0.8           # 80% train, 20% test

# --- MULTI-TASK LOSS WEIGHTS (Must sum to 1.0 for balanced training) ---
ALPHA_EMOTION = 0.6         # Weight for Emotion Loss
BETA_SENTIMENT = 0.4        # Weight for Sentiment Loss
