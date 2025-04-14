import torch
import torchvision.transforms as tf
import torchvision.transforms.functional as F

# --- Tiling Transform --- 
def tile_image(img_tensor):
    """ Takes a CHW image tensor (e.g., 1x28x28) and returns a tiled 2x2 version. """
    C, H, W = img_tensor.shape
    if H != 28 or W != 28:
        # Optional: Add handling or warning if input is not 28x28
        print(f"Warning: tile_image expected 28x28 input, got {H}x{W}")
        return img_tensor # Or raise error

    # Resize to half size (14x14)
    # Antialias=True is recommended for downsampling
    small_img = F.resize(img_tensor, [H // 2, W // 2], interpolation=F.InterpolationMode.BILINEAR, antialias=True)

    # Create new tensor and tile
    tiled_img = torch.zeros_like(img_tensor)
    tiled_img[:, :H//2, :W//2] = small_img  # Top-left
    tiled_img[:, :H//2, W//2:] = small_img  # Top-right
    tiled_img[:, H//2:, :W//2] = small_img  # Bottom-left
    tiled_img[:, H//2:, W//2:] = small_img  # Bottom-right

    return tiled_img

# Modify transform pipeline to include tiling
img_tile_train_transform = tf.Compose([
    tf.ToTensor(), # Convert PIL Image to tensor first (0-1 range)
    tf.Lambda(tile_image), # Apply the tiling
    tf.Lambda(lambda t: (t * 2) - 1) # Normalize to -1 to 1 AFTER tiling
])
# --- End Tiling Transform --- 