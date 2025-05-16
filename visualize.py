import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from model import AudioClassifier
from preprocess import AudioFeatureExtractor, AudioDataset, create_dataloaders
import shap
import librosa
import random
from pathlib import Path
import argparse


def visualize_dataset_distribution(data_dir):
    """Visualize the distribution of real and fake audio samples in the dataset."""
    real_dir = os.path.join(data_dir, 'real')
    fake_dir = os.path.join(data_dir, 'fake')
    
    real_count = len([f for f in os.listdir(real_dir) if f.endswith(('.wav', '.mp3'))]) if os.path.exists(real_dir) else 0
    fake_count = len([f for f in os.listdir(fake_dir) if f.endswith(('.wav', '.mp3'))]) if os.path.exists(fake_dir) else 0
    
    # Create distribution plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(['Real', 'Fake'], [real_count, fake_count], color=['green', 'red'])
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height}', ha='center', va='bottom')
    
    plt.title('Distribution of Audio Samples in Dataset')
    plt.ylabel('Number of Samples')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add percentage labels
    total = real_count + fake_count
    real_pct = (real_count / total) * 100 if total > 0 else 0
    fake_pct = (fake_count / total) * 100 if total > 0 else 0
    
    plt.annotate(f'{real_pct:.1f}%', xy=(0, real_count/2), ha='center', va='center')
    plt.annotate(f'{fake_pct:.1f}%', xy=(1, fake_count/2), ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('dataset_distribution.png')
    plt.close()
    
    print(f"Dataset distribution visualization saved as 'dataset_distribution.png'")
    return {
        'real_count': real_count,
        'fake_count': fake_count,
        'total': total,
        'real_percentage': real_pct,
        'fake_percentage': fake_pct
    }


def visualize_spectrograms(data_dir, feature_extractor, num_samples=3):
    """Visualize mel spectrograms for random samples from each class."""
    # Get random samples from each class
    real_dir = os.path.join(data_dir, 'real')
    fake_dir = os.path.join(data_dir, 'fake')
    
    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) 
                 if f.endswith(('.wav', '.mp3'))] if os.path.exists(real_dir) else []
    fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) 
                 if f.endswith(('.wav', '.mp3'))] if os.path.exists(fake_dir) else []
    
    # Select random samples (or all if fewer than num_samples)
    real_samples = random.sample(real_files, min(num_samples, len(real_files))) if real_files else []
    fake_samples = random.sample(fake_files, min(num_samples, len(fake_files))) if fake_files else []
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 8))
    
    # Plot real samples
    for i, sample_path in enumerate(real_samples):
        features = feature_extractor.extract_features(sample_path)
        if features is not None:
            im = axes[0, i].imshow(features, aspect='auto', origin='lower', cmap='viridis')
            axes[0, i].set_title(f'Real Sample {i+1}')
            axes[0, i].set_xlabel('Time')
            axes[0, i].set_ylabel('Mel Frequency')
            plt.colorbar(im, ax=axes[0, i], format='%+2.0f dB')
    
    # Plot fake samples
    for i, sample_path in enumerate(fake_samples):
        features = feature_extractor.extract_features(sample_path)
        if features is not None:
            im = axes[1, i].imshow(features, aspect='auto', origin='lower', cmap='viridis')
            axes[1, i].set_title(f'Fake Sample {i+1}')
            axes[1, i].set_xlabel('Time')
            axes[1, i].set_ylabel('Mel Frequency')
            plt.colorbar(im, ax=axes[1, i], format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig('mel_spectrograms_comparison.png')
    plt.close()
    
    print(f"Mel spectrogram visualization saved as 'mel_spectrograms_comparison.png'")


def visualize_model_performance(data_dir, batch_size=32):
    """Visualize model performance metrics including accuracy and confusion matrix."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    feature_extractor = AudioFeatureExtractor()
    
    # Load the trained model
    classifier = AudioClassifier(device)
    
    if os.path.exists('audio_classifier.pth'):
        checkpoint = torch.load('audio_classifier.pth', map_location=device)
        classifier.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded successfully with accuracy: {checkpoint['accuracy']:.2f}%")
    else:
        print("No trained model found. Please train the model first.")
        return
    
    # Create dataloaders
    _, test_loader = create_dataloaders(data_dir, feature_extractor, batch_size)
    
    # Evaluate model
    classifier.model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = classifier.model(data)
            _, predicted = torch.max(output.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], 
                yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Calculate and plot classification report
    report = classification_report(all_targets, all_preds, 
                                  target_names=['Real', 'Fake'], 
                                  output_dict=True)
    
    # Convert to DataFrame for easier visualization
    report_df = pd.DataFrame(report).transpose()
    
    # Plot classification metrics
    plt.figure(figsize=(12, 6))
    
    # Plot precision, recall, f1-score for each class
    metrics = ['precision', 'recall', 'f1-score']
    x = np.arange(len(metrics))
    width = 0.35
    
    real_metrics = [report['Real'][m] for m in metrics]
    fake_metrics = [report['Fake'][m] for m in metrics]
    
    plt.bar(x - width/2, real_metrics, width, label='Real')
    plt.bar(x + width/2, fake_metrics, width, label='Fake')
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Classification Metrics by Class')
    plt.xticks(x, metrics)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('classification_metrics.png')
    plt.close()
    
    # Plot accuracy
    accuracy = report_df.loc['accuracy', 'precision']
    plt.figure(figsize=(8, 6))
    plt.bar(['Model Accuracy'], [accuracy], color='blue')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.text(0, accuracy + 0.02, f'{accuracy:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('model_accuracy.png')
    plt.close()
    
    print(f"Model performance visualizations saved as 'confusion_matrix.png', 'classification_metrics.png', and 'model_accuracy.png'")
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report
    }


def visualize_feature_importance(model, feature_extractor, audio_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Visualize feature importance using SHAP values."""
    # Extract features from the audio file
    features = feature_extractor.extract_features(audio_path)
    if features is None:
        print(f"Error extracting features from {audio_path}")
        return None
    
    # Prepare input for the model
    features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
    features_tensor = features_tensor.to(device)
    
    # Create background dataset for SHAP
    background = features_tensor.clone()
    background = background.repeat(10, 1, 1, 1)  # Use 10 samples for background
    
    # Create SHAP explainer
    model.eval()
    explainer = shap.DeepExplainer(model, background)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(features_tensor)
    
    # Plot SHAP values
    plt.figure(figsize=(12, 10))
    
    # Get prediction
    with torch.no_grad():
        output = model(features_tensor)
        pred_class = torch.exp(output).max(1)[1].item()
        confidence = torch.exp(output).max(1)[0].item()
    
    # Plot original spectrogram
    plt.subplot(2, 1, 1)
    plt.imshow(features, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel Spectrogram - Predicted as {"Fake" if pred_class == 1 else "Real"} with {confidence*100:.2f}% confidence')
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')
    
    # Plot SHAP values
    plt.subplot(2, 1, 2)
    if isinstance(shap_values, list):
        shap_data = shap_values[pred_class].squeeze()
    else:
        shap_data = shap_values.squeeze()
    
    # Ensure proper reshaping for visualization
    if len(shap_data.shape) > 2:
        shap_data = shap_data.mean(axis=0)  # Average across channels if present
    
    plt.imshow(shap_data, aspect='auto', origin='lower', cmap='RdBu_r')
    plt.colorbar(label='SHAP Value')
    plt.title('SHAP Values - Feature Importance')
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    print(f"Feature importance visualization saved as 'feature_importance.png'")
    
    return {
        'prediction': 'fake' if pred_class == 1 else 'real',
        'confidence': confidence,
        'shap_values': shap_values
    }


def visualize_waveforms(data_dir, num_samples=3):
    """Visualize waveforms of audio samples from each class."""
    # Get random samples from each class
    real_dir = os.path.join(data_dir, 'real')
    fake_dir = os.path.join(data_dir, 'fake')
    
    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) 
                 if f.endswith(('.wav', '.mp3'))] if os.path.exists(real_dir) else []
    fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) 
                 if f.endswith(('.wav', '.mp3'))] if os.path.exists(fake_dir) else []
    
    # Select random samples
    real_samples = random.sample(real_files, min(num_samples, len(real_files))) if real_files else []
    fake_samples = random.sample(fake_files, min(num_samples, len(fake_files))) if fake_files else []
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 8))
    
    # Plot real samples
    for i, sample_path in enumerate(real_samples):
        try:
            y, sr = librosa.load(sample_path, sr=22050, duration=3)
            axes[0, i].plot(np.linspace(0, len(y)/sr, len(y)), y)
            axes[0, i].set_title(f'Real Sample {i+1} Waveform')
            axes[0, i].set_xlabel('Time (s)')
            axes[0, i].set_ylabel('Amplitude')
        except Exception as e:
            print(f"Error loading {sample_path}: {str(e)}")
    
    # Plot fake samples
    for i, sample_path in enumerate(fake_samples):
        try:
            y, sr = librosa.load(sample_path, sr=22050, duration=3)
            axes[1, i].plot(np.linspace(0, len(y)/sr, len(y)), y)
            axes[1, i].set_title(f'Fake Sample {i+1} Waveform')
            axes[1, i].set_xlabel('Time (s)')
            axes[1, i].set_ylabel('Amplitude')
        except Exception as e:
            print(f"Error loading {sample_path}: {str(e)}")
    
    plt.tight_layout()
    plt.savefig('waveforms_comparison.png')
    plt.close()
    
    print(f"Waveform visualization saved as 'waveforms_comparison.png'")


def main():
    """Run all visualizations."""
    parser = argparse.ArgumentParser(description='Generate visualizations for audio deepfake detection')
    parser.add_argument('--data_dir', type=str, default='dataset', help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for model evaluation')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of samples to visualize per class')
    parser.add_argument('--sample_path', type=str, default=None, help='Path to a specific audio sample for feature importance visualization')
    args = parser.parse_args()
    
    # Initialize feature extractor
    feature_extractor = AudioFeatureExtractor()
    
    # Create output directory for visualizations
    os.makedirs('visualizations', exist_ok=True)
    
    # Visualize dataset distribution
    print("\n1. Visualizing dataset distribution...")
    stats = visualize_dataset_distribution(args.data_dir)
    print(f"Dataset stats: {stats['real_count']} real samples, {stats['fake_count']} fake samples")
    
    # Visualize spectrograms
    print("\n2. Visualizing mel spectrograms...")
    visualize_spectrograms(args.data_dir, feature_extractor, args.num_samples)
    
    # Visualize waveforms
    print("\n3. Visualizing waveforms...")
    visualize_waveforms(args.data_dir, args.num_samples)
    
    # Visualize model performance
    print("\n4. Visualizing model performance...")
    performance = visualize_model_performance(args.data_dir, args.batch_size)
    if performance:
        print(f"Model accuracy: {performance['accuracy']:.2f}")
    
    # Visualize feature importance for a specific sample
    if args.sample_path and os.path.exists(args.sample_path):
        print(f"\n5. Visualizing feature importance for {args.sample_path}...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        classifier = AudioClassifier(device)
        
        if os.path.exists('audio_classifier.pth'):
            checkpoint = torch.load('audio_classifier.pth', map_location=device)
            classifier.model.load_state_dict(checkpoint['model_state_dict'])
            result = visualize_feature_importance(classifier.model, feature_extractor, args.sample_path, device)
            if result:
                print(f"Prediction: {result['prediction']} with {result['confidence']*100:.2f}% confidence")
        else:
            print("No trained model found. Please train the model first.")
    else:
        # Use a sample from the dataset
        fake_dir = os.path.join(args.data_dir, 'fake')
        if os.path.exists(fake_dir):
            fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) 
                         if f.endswith(('.wav', '.mp3'))]
            if fake_files:
                sample_path = fake_files[0]
                print(f"\n5. Visualizing feature importance for {sample_path}...")
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                classifier = AudioClassifier(device)
                
                if os.path.exists('audio_classifier.pth'):
                    checkpoint = torch.load('audio_classifier.pth', map_location=device)
                    classifier.model.load_state_dict(checkpoint['model_state_dict'])
                    result = visualize_feature_importance(classifier.model, feature_extractor, sample_path, device)
                    if result:
                        print(f"Prediction: {result['prediction']} with {result['confidence']*100:.2f}% confidence")
                else:
                    print("No trained model found. Please train the model first.")
    
    print("\nAll visualizations completed successfully!")
    print("Generated visualization files:")
    print("- dataset_distribution.png")
    print("- mel_spectrograms_comparison.png")
    print("- waveforms_comparison.png")
    print("- confusion_matrix.png")
    print("- classification_metrics.png")
    print("- model_accuracy.png")
    print("- feature_importance.png")


if __name__ == '__main__':
    main()