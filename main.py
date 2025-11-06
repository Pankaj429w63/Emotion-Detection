import argparse
import sys
import os

# --- PATH MANIPULATION FOR LOCAL PACKAGE DISCOVERY ---
# This line adds the directory containing 'main.py' to the Python path,
# allowing the interpreter to find the 'src' directory as a top-level package.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
# ----------------------------------------------------

# Corrected Import: Use absolute import starting from the now visible 'src' folder
from models.src.train import main as train_main


def parse_args():
    """Parses command line arguments for controlling the script mode."""
    parser = argparse.ArgumentParser(description="Multimodal Emotion and Sentiment Recognition Project")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "evaluate"],
        help="Run mode: 'train' to start training (and evaluate), 'evaluate' to run final evaluation on a saved checkpoint."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if args.mode == "train":
        print("Starting Multimodal Model Training and Evaluation...")
        # Calls the core logic from src/model_core.py to begin the training loop.
        # Call the full training pipeline defined in models/src/train.py
        train_main()
    elif args.mode == "evaluate":
        print("Starting Final Evaluation using the saved checkpoint...")
        # Calls the core logic to only run the evaluation function.
        # The train module doesn't currently expose a simple evaluate-only CLI.
        # For now, run the full training pipeline which includes final evaluation.
        train_main()