import torch
import torch.nn as nn
import torchvision.models as models

class SmallCNN(nn.Module):
    def __init__(self, feature_dim=256):
        super(SmallCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2), # (32, 32, 128)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # (32, 16, 64)

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # (64, 8, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # (64, 4, 16)

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # (128, 2, 8)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), # (128, 1, 1)
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

class VideoTTCPredictor(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.2, action_dim=16, use_actions=True, num_layers=2, backbone_type="custom"):
        super(VideoTTCPredictor, self).__init__()
        self.use_actions = use_actions
        self.backbone_type = backbone_type
        
        # 1. Feature extractor selection
        if self.backbone_type == "resnet18":
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
            cnn_out_dim = 512
        else:
            self.feature_extractor = SmallCNN(feature_dim=256)
            cnn_out_dim = 256
        
        if self.use_actions and action_dim > 0:
            # Action embedding layer (5 discrete actions in dataset: 0, 1, 2, 3, 4)
            self.action_embedding = nn.Embedding(5, action_dim)
            lstm_input_size = cnn_out_dim + action_dim
        else:
            self.use_actions = False
            lstm_input_size = cnn_out_dim
        
        # LSTM input size is CNN feature size + optional action embedding size
        self.lstm = nn.LSTM(
            input_size=lstm_input_size, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if (dropout > 0 and num_layers > 1) else 0
        )
        
        # 2. Regression head to output the continuous TTC value
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1)
        )
        
    def forward(self, x, actions=None):
        # Expected input shape: 
        # x: (batch_size, sequence_length, channels, height, width)
        # actions: (batch_size, sequence_length) or None
        batch_size, seq_len, c, h, w = x.shape
        
        # Collapse batch and sequence dimensions to pass frames through CNN simultaneously
        x = x.view(batch_size * seq_len, c, h, w)
        spatial_features = self.feature_extractor(x)
        spatial_features = spatial_features.view(batch_size, seq_len, -1)
        
        if self.use_actions and actions is not None:
            # Embed actions and shape to (batch_size, seq_len, action_dim)
            action_feats = self.action_embedding(actions)
            # Concatenate spatial features and action features
            combined_feats = torch.cat([spatial_features, action_feats], dim=-1)
        else:
            combined_feats = spatial_features
        
        # Pass spatial sequence through LSTM
        lstm_out, _ = self.lstm(combined_feats)
        
        # We only care about the temporal context at the VERY LAST frame in the sequence
        last_frame_features = lstm_out[:, -1, :]
        
        return self.regressor(last_frame_features)