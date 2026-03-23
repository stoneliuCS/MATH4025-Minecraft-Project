import torch

import torch.nn as nn


class LSTMConvModel(nn.Module):
    def __init__(self, input_channels=1, hidden_size=128, num_layers=2, output_size=10):
        super(LSTMConvModel, self).__init__()
        
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
        # After conv layers: 64x64 -> 32x32 -> 16x16
        # Flattened size: 64 * 16 * 16 = 16384
        self.conv_output_size = 64 * 16 * 16
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.conv_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, channels, height, width)
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Process each frame through conv layers
        x = x.view(batch_size * seq_len, x.size(2), x.size(3), x.size(4))
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        # Flatten and reshape for LSTM
        x = x.view(batch_size, seq_len, -1)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last output for classification
        x = self.fc(lstm_out[:, -1, :])
        
        return x