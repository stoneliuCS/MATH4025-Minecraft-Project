import torch
import torch.nn as nn
import numpy as np

class VisionEncoder(nn.Module):
    """
    Simple CNN to extract visual features from Minecraft POV.
    Takes 64x64x3 RGB image, outputs 64-dimensional feature vector.
    """
    def __init__(self, output_dim=64):
        super().__init__()
        
        # Convolutional layers to process image
        self.conv = nn.Sequential(
            # Input: 64x64x3 (H, W, C)
            nn.Conv2d(3, 16, kernel_size=8, stride=4, padding=0),  # -> 15x15x16
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),  # -> 6x6x32
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),  # -> 4x4x32
            nn.ReLU(),
            nn.Flatten(),  # -> 512 (4*4*32)
        )
        
        # Fully connected layer to compress to desired size
        self.fc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, pov):
        """
        Args:
            pov: numpy array or tensor of shape (batch, H, W, C) or (H, W, C)
                 Values should be in range [0, 255]
        Returns:
            features: tensor of shape (batch, output_dim) or (output_dim,)
        """
        # Convert to torch tensor if needed
        if isinstance(pov, np.ndarray):
            pov = torch.from_numpy(pov)
        
        # Add batch dimension if needed
        if pov.dim() == 3:
            pov = pov.unsqueeze(0)  # (H, W, C) -> (1, H, W, C)
        
        # Convert from (batch, H, W, C) to (batch, C, H, W)
        pov = pov.permute(0, 3, 1, 2).float()
        
        # Normalize to [0, 1]
        pov = pov / 255.0
        
        # Pass through network
        conv_features = self.conv(pov)
        features = self.fc(conv_features)
        
        # Remove batch dimension if input was single image
        if features.shape[0] == 1:
            features = features.squeeze(0)
        
        return features


def test_vision_model():
    """Test that the vision model works correctly."""
    print("Testing VisionEncoder...")
    
    # Create model
    model = VisionEncoder(output_dim=64)
    
    # Create fake POV image (64x64x3)
    fake_pov = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
    
    # Test single image
    features = model(fake_pov)
    print(f"Single image input shape: {fake_pov.shape}")
    print(f"Output features shape: {features.shape}")
    print(f"Features (first 10): {features[:10]}")
    
    # Test batch of images
    batch_pov = np.random.randint(0, 256, size=(4, 64, 64, 3), dtype=np.uint8)
    batch_features = model(batch_pov)
    print(f"\nBatch input shape: {batch_pov.shape}")
    print(f"Batch output shape: {batch_features.shape}")
    
    print("\n✓ VisionEncoder test passed!")


if __name__ == "__main__":
    test_vision_model()