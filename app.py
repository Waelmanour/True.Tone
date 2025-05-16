import os
import sys
from pathlib import Path

# Add the current directory to Python path before any other imports
sys.path.append(str(Path(__file__).parent.absolute()))

# Initialize asyncio event loop
import asyncio

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

# Import required packages
import streamlit as st
import torch
import torch.backends.cudnn as cudnn
import librosa
import numpy as np
import matplotlib.pyplot as plt
import shap

# Import local modules after environment setup
from model import AudioClassifier
from preprocess import AudioFeatureExtractor

# Configure PyTorch settings
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# Configure PyTorch settings
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# Set page config
st.set_page_config(
    page_title="Audio Deepfake Detector Made By Wael Mansour",
    page_icon="🎵",
    layout="wide"
)

# Initialize model and feature extractor
def load_model():
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
            from main import train_model
            print("Training new model since no existing model found")
            dataset_dir = 'dataset'
            classifier, feature_extractor = train_model(dataset_dir, batch_size=32, epochs=5)
    except Exception as e:
        print(f"Error loading/training model: {str(e)}")
        return None, None
    
    return classifier, feature_extractor

# Analyze audio function
def analyze_audio(classifier, feature_extractor, audio_file):
    # Save uploaded file temporarily
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.getbuffer())
    
    # Extract features
    features = feature_extractor.extract_features("temp_audio.wav")
    if features is None:
        return None
    
    # Prepare input for the model
    features = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
    features = features.to(classifier.device)
    
    # Get activation maps from the last convolutional layer
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # Register hook to get activation from last conv layer
    classifier.model.conv3.register_forward_hook(get_activation('conv3'))
    
    # Make prediction
    with torch.no_grad():
        output = classifier.model(features)
        prediction = torch.exp(output).max(1)[1].item()
        confidence = torch.exp(output).max(1)[0].item()
    
    # Get activation map
    activation_map = activation['conv3'].squeeze().mean(dim=0).cpu().numpy()
    
    # Clean up
    os.remove("temp_audio.wav")
    
    return {
        'prediction': 'fake' if prediction == 1 else 'real',
        'confidence': confidence,
        'features': features.squeeze().cpu().numpy(),
        'activation_map': activation_map
    }

# Main app
def main():
    st.title("🎵 Audio Deepfake Detector")
    st.write("Upload an audio file to check if it's real or fake")
    st.write("Wael Mansour, Mostafa Salem, Mohamadi Alaa, Mohammed Hany, Yousef El Araby")
    # Load model
    classifier, feature_extractor = load_model()
    if classifier is None:
        return
    
    # File uploader
    audio_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3'])
    
    if audio_file is not None:
        st.audio(audio_file)
        
        with st.spinner('Analyzing audio...'):
            result = analyze_audio(classifier, feature_extractor, audio_file)
        
        if result:
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Detection Result")
                prediction_color = "green" if result['prediction'] == 'real' else "red"
                st.markdown(f"<h3 style='color: {prediction_color};'>{result['prediction'].upper()}</h3>", 
                          unsafe_allow_html=True)
                st.metric("Confidence", f"{result['confidence']*100:.2f}%")
            
            with col2:
                st.subheader("Feature Importance Visualization")
                try:
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                    
                    # Plot mel spectrogram
                    im1 = ax1.imshow(result['features'], aspect='auto', origin='lower')
                    ax1.set_title('Mel Spectrogram')
                    plt.colorbar(im1, ax=ax1)
                    
                    # Plot activation heatmap
                    im2 = ax2.imshow(result['activation_map'], aspect='auto', origin='lower', cmap='hot')
                    ax2.set_title('Activation Heatmap')
                    plt.colorbar(im2, ax=ax2)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.error(f"Error generating visualization: {str(e)}")
                    return
            
            st.markdown("""---
            ### Understanding the Results
            - **Real**: The audio appears to be authentic
            - **Fake**: The audio shows signs of being artificially generated
            - **Mel Spectrogram**: Shows the frequency content of the audio over time
            - **Activation Heatmap**: Highlights the regions that most influenced the model's decision
            """)

if __name__ == '__main__':
    main()