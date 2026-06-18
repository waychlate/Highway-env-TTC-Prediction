import torch
import torch.nn as nn
import torchvision.models as models

class VideoTTCPredictor(nn.Module):
    def __init__(self, hidden_dim=256):
        super(VideoTTCPredictor, self).__init__()
        
        # 1. Use a proven backbone (ResNet18) stripped of its final classification layer
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freezing early layers can speed up training if data is limited
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        
        # 2. Regression head to output the continuous TTC value
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        # Expected input shape: (batch_size, sequence_length, channels, height, width)
        batch_size, seq_len, c, h, w = x.shape
        
        # Collapse batch and sequence dimensions to pass frames through CNN simultaneously
        x = x.view(batch_size * seq_len, c, h, w)
        spatial_features = self.feature_extractor(x) # Shape: (batch_size * seq_len, 512, 1, 1)
        spatial_features = spatial_features.view(batch_size, seq_len, -1) # Shape: (batch_size, seq_len, 512)
        
        # Pass spatial sequence through LSTM
        lstm_out, _ = self.lstm(spatial_features) # Shape: (batch_size, seq_len, hidden_dim)
        
        # We only care about the temporal context at the VERY LAST frame in the sequence
        last_frame_features = lstm_out[:, -1, :]
        
        return self.regressor(last_frame_features)