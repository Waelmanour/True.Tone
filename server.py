import os
import secrets
from pathlib import Path
import sys

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent.absolute()))

# Initialize asyncio event loop
import asyncio
import librosa  # For audio processing, especially MP3 files

def init_event_loop():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

# Initialize the event loop before importing torch
init_event_loop()

# Initialize PyTorch and CUDA before importing other modules
os.environ['PYTORCH_JIT'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import torch
import torch.backends.cudnn as cudnn
import tempfile
import numpy as np

# Import local modules
from model import AudioClassifier
from preprocess import AudioFeatureExtractor

# Configure PyTorch settings
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
class Config:
    SECRET_KEY = secrets.token_hex(16)
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    ALLOWED_EXTENSIONS = {'wav', 'mp3'}
    API_KEYS = {"test_key": "admin"}  # In production, use a proper database

app.config.from_object(Config)

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model and feature extractor
classifier = None
feature_extractor = None

def load_model():
    global classifier, feature_extractor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    classifier = AudioClassifier(device)
    feature_extractor = AudioFeatureExtractor()
    
    try:
        if os.path.exists('audio_classifier.pth'):
            checkpoint = torch.load('audio_classifier.pth', map_location=device)
            classifier.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded successfully with accuracy: {checkpoint['accuracy']:.2f}%")
            classifier.model.eval()
        else:
            print("Error: Model file not found")
            return False
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False
    
    return True

# Helper functions
def allowed_file(filename):
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    # Check if the extension is in allowed extensions
    print(f"Checking file extension: {ext} for file {filename}")
    return ext in app.config['ALLOWED_EXTENSIONS']

def verify_api_key():
    api_key = request.headers.get('X-API-Key')
    if not api_key or api_key not in app.config['API_KEYS']:
        return False
    return True

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check API key
    if not verify_api_key():
        return jsonify({"detail": "Invalid or missing API key"}), 401
    
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({"detail": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"detail": "No selected file"}), 400
    
    # Ensure filename is properly secured before checking extension
    filename = secure_filename(file.filename)
    if not allowed_file(filename):
        return jsonify({"detail": f"File type not allowed. Only .wav and .mp3 files are accepted."}), 400
    
    try:
        # Save file to temporary location with appropriate suffix
        # We already secured the filename above
        file_ext = os.path.splitext(filename)[1].lower()  # This returns '.mp3' or '.wav' with the dot
        # Ensure file_ext starts with a dot
        if not file_ext.startswith('.'):
            file_ext = '.' + file_ext
        print(f"Processing file with extension: {file_ext}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name
            print(f"Saved temporary file: {temp_path} with extension {file_ext}")
        
        # Check if file is mp3 and convert to wav if needed
        # No need to extract extension again, reuse the one we already have
        if file_ext.lower() == '.mp3':
            try:
                # Load mp3 file with librosa and save as wav
                y, sr = librosa.load(temp_path, sr=22050, mono=True)
                wav_path = temp_path.replace('.mp3', '.wav')
                import soundfile as sf
                sf.write(wav_path, y, sr)
                # Remove original mp3 temp file
                os.unlink(temp_path)
                temp_path = wav_path
                print(f"Successfully converted MP3 to WAV: {file.filename}")
            except Exception as e:
                os.unlink(temp_path)  # Delete temp file
                print(f"Error converting MP3 to WAV: {str(e)}")
                return jsonify({"detail": f"Failed to convert mp3 to wav: {str(e)}"}), 400
                
        # Extract features
        features = feature_extractor.extract_features(temp_path)
        if features is None:
            os.unlink(temp_path)  # Delete temp file
            return jsonify({"detail": "Failed to extract features from audio"}), 400
        
        # Prepare input for the model
        features = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
        features = features.to(classifier.device)
        
        # Get prediction
        with torch.no_grad():
            output = classifier.model(features)
            probabilities = torch.exp(output)
            pred_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][pred_class].item() * 100
        
        # Map prediction to label
        prediction = "fake" if pred_class == 1 else "real"
        
        # Delete temporary file
        os.unlink(temp_path)
        
        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 2)
        })
        
    except Exception as e:
        return jsonify({"detail": f"Error processing audio: {str(e)}"}), 500

@app.route('/api/status', methods=['GET'])
def api_status():
    return jsonify({"status": "online", "model_loaded": classifier is not None})

@app.route('/api/docs', methods=['GET'])
def api_docs():
    return jsonify({
        "endpoints": [
            {
                "path": "/predict",
                "method": "POST",
                "description": "Analyze audio file for deepfake detection",
                "parameters": [
                    {
                        "name": "X-API-Key",
                        "in": "header",
                        "required": True,
                        "description": "API key for authentication"
                    },
                    {
                        "name": "file",
                        "in": "formData",
                        "required": True,
                        "description": "Audio file (.wav or .mp3) to analyze"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Successful analysis",
                        "schema": {
                            "prediction": "string (real or fake)",
                            "confidence": "number (percentage)"
                        }
                    },
                    "400": {"description": "Bad request"},
                    "401": {"description": "Unauthorized"},
                    "500": {"description": "Server error"}
                }
            },
            {
                "path": "/api/status",
                "method": "GET",
                "description": "Check API status",
                "responses": {
                    "200": {
                        "description": "API status",
                        "schema": {
                            "status": "string",
                            "model_loaded": "boolean"
                        }
                    }
                }
            }
        ]
    })

# Main entry point
if __name__ == '__main__':
    # Load model before starting the server
    if load_model():
        print("Model loaded successfully, starting server...")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("Failed to load model, server not started")