# Emotion Detection (Multimodal Emotion & Sentiment Recognition)

This project implements a **multimodal emotion and sentiment detection pipeline** using **text, audio, and video features**.  
It includes:

- A **training pipeline** (PyTorch)
- A **Flask REST API** for inference
- A **Gradio UI demo**
- Utility scripts for running predictions from the command line

The model is designed around MELD-style emotion classes:

- **Emotions:** Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise  
- **Sentiments:** Positive, Negative, Neutral

---

## 🔧 Tech Stack

- **Language:** Python
- **Core ML:** PyTorch, NumPy, scikit-learn
- **NLP / Transformers:** Hugging Face `transformers`
- **Monitoring:** tensorboardX
- **APIs / UI:** Flask, Gradio
- **Others:** pandas, tqdm

All dependencies are listed in `requirements.txt`.

---

## 📁 Project Structure

```bash
Emotion-Detection/
├── app.py              # Flask API server (POST /predict)
├── main.py             # CLI entry point for training / evaluation
├── predict.py          # Simple CLI-based prediction script
├── gradio_demo.py      # Gradio interface for interactive demo
├── config.py           # Global config, labels, hyperparameters, paths
├── index.html          # Frontend page (can be used with the API)
├── dia446_utt7.mp4     # Sample video clip (example data)
├── dia945_utt12.mp4    # Sample video clip (example data)
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (MODEL_DIR, etc.)
├── LICENSE             # MIT License
├── README.md           # Project documentation (this file)
├── models/             # (Expected) trained models & source code
│   ├── final_model.pt          # Final trained model (for demo/API)
│   ├── meld_fusion_model.pt    # Multimodal MELD fusion checkpoint
│   └── src/
│       ├── train.py            # Training / evaluation pipeline
│       ├── models.py           # Model architectures
│       ├── utils.py            # Helper functions (load_model, etc.)
│       └── ...
└── python/             # (Optional) extra helper scripts / env files



 Installation

Clone the repository:
git clone https://github.com/Pankaj429w63/Emotion-Detection.git
cd Emotion-Detection
Create and activate a virtual environment (recommended):

# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate


Install dependencies:

pip install -r requirements.txt
 Configuration

The project uses a .env file and config.py for configuration.

Key settings (from config.py):

MODEL_SAVE_PATH – path to main checkpoint

MODEL_SAVE_PATH = os.environ.get("MODEL_DIR", "./models/") + "meld_fusion_model.pt"


EMOTION_CLASSES – list of 7 emotions

SENTIMENT_CLASSES – list of 3 sentiments

Feature dimensions (for simulated multimodal features):

TEXT_FEAT_DIM = 768 (e.g., BERT embeddings)

AUDIO_FEAT_DIM = 128

VIDEO_FEAT_DIM = 512

Training hyperparameters: BATCH_SIZE, LEARNING_RATE, EPOCHS, etc.

.env Example

Create a .env file in the project root:

MODEL_DIR=./models/


Make sure your trained model file (e.g., meld_fusion_model.pt or final_model.pt) exists in that directory.

 Training & Evaluation

The main training/evaluation entry point is main.py, which delegates to models/src/train.py.

Train the model
python main.py --mode train


This will:

Initialize the multimodal fusion model

Train on your dataset (configure inside models/src/train.py)

Save the checkpoint in the models/ directory

Evaluate the model
python main.py --mode evaluate


Currently, evaluation is coupled with the training pipeline; --mode evaluate runs the evaluation portion defined in train.py (and may also run training, depending on how train_main is implemented).

 Check models/src/train.py for dataset paths, preprocessing, and training logic.

Quick CLI Prediction (Terminal)

You can run a quick CLI-based prediction using predict.py.

python predict.py


This script:

Loads MultimodalFusionModel from models/src/models.py

Loads weights from ./models/final_model.pt

Simulates audio/video features

Prints the predicted emotion and confidence for example sentences

Sample output (format):

========================================
| INPUT SENTENCE: 'Oh my god, I haven't seen you in ages, this is amazing!'
| PREDICTED EMOTION: JOY
| CONFIDENCE: 92.34%
========================================


To adapt it for custom input, modify the bottom of predict.py or wrap predict_emotion(sentence) in your own script.

 Flask REST API

app.py exposes a simple REST API for emotion + sentiment prediction.

Run the API server
python app.py


By default, it starts on:

http://0.0.0.0:5000/

You’ll see this message at the root:

## Emotion Detection API is running. Use POST /predict to get results.

/predict Endpoint

Method: POST

URL: http://localhost:5000/predict

Body (JSON):

{
  "text": "I am so happy to see you!"
}

Example using curl
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"I am so happy to see you!\"}"

Example Response
{
  "input_text": "I am so happy to see you!",
  "predicted_emotion": "Joy",
  "emotion_confidence": "85.23%",
  "predicted_sentiment": "Positive",
  "sentiment_confidence": "91.47%",
  "model_status": "OK"
}


 Note:

If MODEL_SAVE_PATH does not exist, the API falls back to a DummyModel that returns random predictions.

Once you place a valid trained checkpoint at MODEL_SAVE_PATH, it will load the real model.

🎛 Gradio Demo UI

gradio_demo.py launches a Gradio-based web UI for interactive experimentation.

Run the Gradio app
python gradio_demo.py


What it does:

Loads the trained model from ./models/final_model.pt

Opens a browser window/tab

Lets you type a sentence and returns:

Predicted emotion

Confidence percentage

The UI includes:

Input: Textbox (“Enter a sentence for emotion analysis...”)

Outputs:

Predicted Emotion

Confidence Level

Note: For simplicity, audio/video features are simulated as zero tensors in this demo.

 Multimodal Design (High Level)

The project is structured for multimodal fusion, combining:

Text (BERT-style embeddings, 768-dim)

Audio (e.g., MFCC/Wav2Vec-like features, 128-dim)

Video (e.g., ResNet/FaceNet embeddings, 512-dim)

Key ideas:

Context window over previous utterances via LSTMs (see CONTEXT_WINDOW in config.py)

Multi-task outputs for emotion and sentiment

Loss balancing using:

ALPHA_EMOTION (emotion loss weight)

BETA_SENTIMENT (sentiment loss weight)

 Current Status & To-Dos

 End-to-end training + dummy-data simulation works

 API + Gradio demo wired to checkpoints

 Real audio/video feature extraction not included in this repo

 Data loading and preprocessing must be configured in models/src/train.py and models/src/utils.py

 License

This project is licensed under the MIT License.
See the LICENSE
 file for details.

⚙️ Installation

Clone the repository:
