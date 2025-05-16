import torch
import torch.nn as nn
import torch.optim as optim
import shap
import matplotlib.pyplot as plt
import numpy as np
from model import AudioClassifier
from preprocess import AudioFeatureExtractor, create_dataloaders

def train_model(dataset_dir, batch_size=32, epochs=10, learning_rate=0.001):
    # Initialize feature extractor and create dataloaders
    feature_extractor = AudioFeatureExtractor()
    train_loader, test_loader = create_dataloaders(dataset_dir, feature_extractor, batch_size)
    
    # Initialize model, optimizer, and loss function
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    classifier = AudioClassifier(device)
    optimizer = optim.Adam(classifier.model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    
    # Train the model
    classifier.train(train_loader, optimizer, criterion, epochs)
    
    # Evaluate the model
    accuracy = classifier.evaluate(test_loader)
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    # Save the trained model
    torch.save({
        'model_state_dict': classifier.model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy
    }, 'audio_classifier.pth')
    
    return classifier, feature_extractor

def explain_prediction(model, feature_extractor, audio_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Extract features from the audio file
    features = feature_extractor.extract_features(audio_path)
    if features is None:
        return None
    
    # Print feature shape for debugging
    print(f"Original feature shape: {features.shape}")
    
    # Prepare input for the model - features are already in correct shape (64, 128)
    features = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
    features = features.to(device)
    
    # Create background dataset for SHAP
    background = features.clone()
    background = background.repeat(100, 1, 1, 1)
    
    # Create SHAP explainer and ensure model is in eval mode
    model.eval()
    explainer = shap.DeepExplainer(model, background)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(features)
    
    # Plot SHAP values
    plt.figure(figsize=(10, 6))
    if isinstance(shap_values, list):
        with torch.no_grad():
            output = model(features)
            pred_class = torch.exp(output).max(1)[1].item()
            shap_data = shap_values[pred_class].squeeze()
    else:
        shap_data = shap_values.squeeze()
    
    # Ensure proper reshaping for visualization
    features_np = features.cpu().numpy().squeeze()
    if len(features_np.shape) > 2:
        features_np = features_np.squeeze(0)  # Remove channel dimension if present
    
    if len(shap_data.shape) > 2:
        shap_data = shap_data.mean(axis=0)  # Average across channels if present
    
    shap.image_plot(shap_data, features_np, show=False)
    plt.title('SHAP Values for Audio Features')
    plt.tight_layout()
    plt.savefig('shap_explanation.png')
    plt.close()
    
    # Make prediction
    with torch.no_grad():
        output = model(features)
        prediction = torch.exp(output).max(1)[1].item()
        confidence = torch.exp(output).max(1)[0].item()
    
    return {
        'prediction': 'fake' if prediction == 1 else 'real',
        'confidence': confidence,
        'explanation_plot': 'shap_explanation.png',
        'shap_values': shap_values,
        'features': features
    }

def main():
    # Example usage
    data_dir = 'dataset'  # Directory containing 'real' and 'fake' subdirectories
    
    # Train the model
    print('Training model...')
    classifier, feature_extractor = train_model(data_dir)
    
    # Example of analyzing a specific audio file from the dataset
    audio_path = 'dataset/fake/recording1.wav_norm_mono.wav'
    print('\nAnalyzing audio file...')
    result = explain_prediction(classifier.model, feature_extractor, audio_path, classifier.device)
    
    if result:
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Explanation plot saved as: {result['explanation_plot']}")

if __name__ == '__main__':
    main()