import torch
import torch.nn as nn
import torchvision.models as models

class VideoTTCPredictor(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.2, action_dim=16):
        super(VideoTTCPredictor, self).__init__()
        
        # 1. Use a proven backbone (ResNet18) stripped of its final classification layer
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Action embedding layer (5 discrete actions in dataset: 0, 1, 2, 3, 4)
        self.action_embedding = nn.Embedding(5, action_dim)
        
        # LSTM input size is ResNet feature size (512) + action embedding size
        self.lstm = nn.LSTM(
            input_size=512 + action_dim, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )
        
        # 2. Regression head to output the continuous TTC value
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1)
        )
        
    def forward(self, x, actions):
        # Expected input shape: 
        # x: (batch_size, sequence_length, channels, height, width)
        # actions: (batch_size, sequence_length)
        batch_size, seq_len, c, h, w = x.shape
        
        # Collapse batch and sequence dimensions to pass frames through CNN simultaneously
        x = x.view(batch_size * seq_len, c, h, w)
        spatial_features = self.feature_extractor(x) # Shape: (batch_size * seq_len, 512, 1, 1)
        spatial_features = spatial_features.view(batch_size, seq_len, -1) # Shape: (batch_size, seq_len, 512)
        
        # Embed actions and shape to (batch_size, seq_len, action_dim)
        action_feats = self.action_embedding(actions)
        
        # Concatenate spatial features and action features
        combined_feats = torch.cat([spatial_features, action_feats], dim=-1) # Shape: (batch_size, seq_len, 512 + action_dim)
        
        # Pass spatial sequence through LSTM
        lstm_out, _ = self.lstm(combined_feats) # Shape: (batch_size, seq_len, hidden_dim)
        
        # We only care about the temporal context at the VERY LAST frame in the sequence
        last_frame_features = lstm_out[:, -1, :]
        
        return self.regressor(last_frame_features)