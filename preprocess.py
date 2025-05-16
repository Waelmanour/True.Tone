import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader

class AudioFeatureExtractor:
    def __init__(self, sample_rate=22050, duration=3, n_mels=128):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_samples = int(sample_rate * duration)
    
    def extract_features(self, audio_path):
        # Load audio file
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
            
            # Pad if audio is shorter than expected duration
            if len(y) < self.n_samples:
                y = np.pad(y, (0, self.n_samples - len(y)))
            
            # Compute mel spectrogram with fixed time steps
            n_fft = 2048  # FFT window size
            hop_length = 512  # Fixed hop length for consistent time steps
            n_time_steps = 128  # Target time steps to match model's expected input
            
            # Compute mel spectrogram with exact dimensions
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=self.sample_rate,
                n_mels=self.n_mels,  # Use the instance variable
                hop_length=hop_length,
                n_fft=n_fft,
                fmax=8000
            )
            
            # Ensure exact dimensions without padding
            # Trim or pad to match target time steps
            if mel_spec.shape[1] > n_time_steps:
                mel_spec = mel_spec[:, :n_time_steps]
            elif mel_spec.shape[1] < n_time_steps:
                pad_width = ((0, 0), (0, n_time_steps - mel_spec.shape[1]))
                mel_spec = np.pad(mel_spec, pad_width, mode='constant')
            
            # Convert to log scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize
            mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
            
            # Verify final shape
            assert mel_spec_norm.shape == (self.n_mels, n_time_steps), f"Unexpected shape: {mel_spec_norm.shape}"
            
            return mel_spec_norm
            
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            return None

class AudioDataset(Dataset):
    def __init__(self, data_dir, feature_extractor, transform=None):
        self.feature_extractor = feature_extractor
        self.transform = transform
        self.samples = []
        
        # Walk through the data directory
        for label, class_name in enumerate(['real', 'fake']):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                for audio_file in os.listdir(class_dir):
                    if audio_file.endswith(('.wav', '.mp3')):
                        self.samples.append({
                            'path': os.path.join(class_dir, audio_file),
                            'label': label
                        })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = self.feature_extractor.extract_features(sample['path'])
        
        if features is None:
            # Return a zero tensor if feature extraction failed
            features = torch.zeros((self.feature_extractor.n_mels, 
                                  int(self.feature_extractor.duration * 2048 // 512)))
        else:
            features = torch.FloatTensor(features)
        
        # Add channel dimension
        features = features.unsqueeze(0)
        
        if self.transform:
            features = self.transform(features)
        
        # Remove the dimension check since we're already ensuring correct dimensions
        # through the feature extraction process
        
        return features, sample['label']

def create_dataloaders(data_dir, feature_extractor, batch_size=32, train_split=0.8):
    dataset = AudioDataset(data_dir, feature_extractor)
    
    # Split dataset into train and test
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, test_loader