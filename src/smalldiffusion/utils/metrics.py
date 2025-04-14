import torch
import torch.nn.functional as F

def calculate_quadrant_mse(image_batch: torch.Tensor) -> float:
    """Calculates the average MSE between the four quadrants of each image in a batch.

    Args:
        image_batch: Tensor of shape [B, C, H, W], where H and W are assumed to be even.

    Returns:
        Average MSE across all images and quadrant pairs in the batch.
    """
    B, C, H, W = image_batch.shape
    if H % 2 != 0 or W % 2 != 0:
        raise ValueError(f"Image dimensions must be even for quadrant splitting, got {H}x{W}")

    H_half, W_half = H // 2, W // 2

    # Extract quadrants
    tl = image_batch[:, :, :H_half, :W_half]
    tr = image_batch[:, :, :H_half, W_half:]
    bl = image_batch[:, :, H_half:, :W_half]
    br = image_batch[:, :, H_half:, W_half:]

    # Calculate MSE between relevant pairs (e.g., comparing adjacent and diagonal)
    # Reduce across spatial and channel dims, keep batch dim
    mse_tl_tr = F.mse_loss(tl, tr, reduction='none').mean(dim=[1, 2, 3])
    mse_tl_bl = F.mse_loss(tl, bl, reduction='none').mean(dim=[1, 2, 3])
    mse_tl_br = F.mse_loss(tl, br, reduction='none').mean(dim=[1, 2, 3]) # Diagonal comparison
    mse_tr_br = F.mse_loss(tr, br, reduction='none').mean(dim=[1, 2, 3])
    mse_bl_br = F.mse_loss(bl, br, reduction='none').mean(dim=[1, 2, 3])
    mse_tr_bl = F.mse_loss(tr, bl, reduction='none').mean(dim=[1, 2, 3]) # Other diagonal

    # Average MSE across all pairs and batches
    # Stack the per-image MSEs and calculate the overall mean
    all_mses = torch.stack([mse_tl_tr, mse_tl_bl, mse_tl_br, mse_tr_br, mse_bl_br, mse_tr_bl], dim=0)
    avg_mse = all_mses.mean()

    return avg_mse.item() 