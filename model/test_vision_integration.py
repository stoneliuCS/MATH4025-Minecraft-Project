import numpy as np
from vision_model import VisionEncoder

def test_vision_with_fake_minecraft_obs():
    """Test vision model with fake Minecraft-like observations."""
    print("Testing vision integration...")
    
    model = VisionEncoder(output_dim=64)
    
    # Simulate different scenarios
    scenarios = {
        "red_wool_visible": np.zeros((64, 64, 3), dtype=np.uint8),
        "grass_only": np.zeros((64, 64, 3), dtype=np.uint8),
        "tree_visible": np.zeros((64, 64, 3), dtype=np.uint8),
    }
    
    # Red wool: red pixels in center
    scenarios["red_wool_visible"][28:36, 28:36, 0] = 200  # Red channel
    
    # Grass: green everywhere
    scenarios["grass_only"][:, :, 1] = 100  # Green channel
    
    # Tree: brown bottom, green top
    scenarios["tree_visible"][40:64, :, :] = [139, 69, 19]  # Brown
    scenarios["tree_visible"][0:40, :, :] = [34, 139, 34]  # Green
    
    # Extract features for each
    print("\nVisual buckets for different scenarios:")
    for name, pov in scenarios.items():
        features = model(pov)
        # FIX: Convert to numpy properly
        features_np = features.detach().numpy()
        bucket = int(np.argmax(features_np[:8]))
        print(f"  {name:20s} -> bucket {bucket}, feature strength: {features_np[bucket]:.2f}")
    
    print("\n✓ Vision integration test complete!")

if __name__ == "__main__":
    test_vision_with_fake_minecraft_obs()