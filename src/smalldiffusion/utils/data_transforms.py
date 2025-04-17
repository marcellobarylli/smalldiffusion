import torch
import torchvision.transforms as tf
import torchvision.transforms.functional as F

# --- Existing Tiling Transform (Downsample -> Tile 2x2 within original size) --- 
def tile_image(img_tensor):
    """ Takes a CHW image tensor (e.g., 1x28x28) and returns a tiled 2x2 version *of a downsampled image within the original size*. """
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

# Transform pipeline for the above tiling
img_tile_train_transform = tf.Compose([
    tf.ToTensor(), # Convert PIL Image to tensor first (0-1 range)
    tf.Lambda(tile_image), # Apply the downsample-tiling
    tf.Lambda(lambda t: (t * 2) - 1) # Normalize to -1 to 1 AFTER tiling
])
# --- End Existing Tiling Transform ---

# --- New Tiling Transform (Repeat 2x2 -> Double size) ---
def tile_image_2x2(img_tensor):
    """ 
    Takes a CHW (3D) or BCHW (4D) image tensor and returns a 2x larger tensor 
    by tiling the input 2x2 along the H and W dimensions.
    """
    # Simple repeat for 2x2 tiling
    if img_tensor.dim() == 3: # Input is [C, H, W]
        # Repeat C=1, H=2, W=2
        tiled_img = img_tensor.repeat(1, 2, 2) 
    elif img_tensor.dim() == 4: # Input is [B, C, H, W]
        # Repeat B=1, C=1, H=2, W=2
        tiled_img = img_tensor.repeat(1, 1, 2, 2)
    else:
        raise ValueError(f"tile_image_2x2 expects a 3D or 4D tensor, got {img_tensor.dim()}D")
    
    return tiled_img

# Transform pipeline for the new 2x2 repeat tiling
img_tile_2x2_transform = tf.Compose([
    tf.ToTensor(), # Convert PIL Image to tensor first (0-1 range)
    tf.Lambda(tile_image_2x2), # Apply the 2x2 repeat tiling
    tf.Lambda(lambda t: (t * 2) - 1) # Normalize to -1 to 1 AFTER tiling
])
# --- End New Tiling Transform --- 