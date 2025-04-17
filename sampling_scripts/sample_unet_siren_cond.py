import torch
from torchvision.utils import make_grid, save_image
import sys
import os
import argparse
from accelerate import Accelerator # Minimal accelerator for device handling

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(1, src_path)

# Imports from smalldiffusion
from smalldiffusion.diffusion import ScheduleLogLinear
from smalldiffusion.data import img_normalize # For saving samples
from smalldiffusion.model import Scaled
from smalldiffusion.model_unet import Unet
from siren_model import AutoDecoder

# --- Conditional Sampling Function (Copied from training script) ---
@torch.no_grad()
def conditional_samples(model, schedule, siren_model, target_labels, batchsize, accelerator):
    """Simplified conditional sampling loop."""
    model.eval()
    siren_model.eval()

    sigmas = schedule.sample_sigmas(20) # Use a fixed number of steps for sampling
    # gam = 1.6 # Example gamma, adjust if needed - Simple Euler for now

    # Generate SIREN conditioning for the target labels
    target_labels = target_labels.to(accelerator.device)
    with torch.no_grad():
        siren_cond = siren_model(target_labels)
        # Repeat condition for each sample in the batch if needed
        num_target_labels = len(target_labels)
        if batchsize % num_target_labels != 0:
            raise ValueError("Batch size must be a multiple of the number of target labels for repeating condition.")
        repeats = batchsize // num_target_labels
        siren_cond = siren_cond.repeat(repeats, 1, 1, 1) # Shape [B, 1, H, W]

    img_channels = 1 # MNIST is grayscale
    # Infer image size from model if possible, otherwise use arg
    # Assuming model has input_dims attribute after loading
    # We need to load the model first to get input_dims accurately.
    # Let's pass image_size as an argument for now.
    noise_shape = (batchsize, img_channels, args.image_size, args.image_size)
    device = accelerator.device

    # Initial noise
    xt = torch.randn(noise_shape, device=device) * sigmas[0] # Generate noise with 1 channel

    for sig1, sig2 in zip(sigmas[:-1], sigmas[1:]):
        sigma = sig1 * torch.ones(batchsize, device=device)

        # Prepare input: noisy image + siren condition
        unet_input = torch.cat([xt, siren_cond], dim=1)

        # Predict noise/x0 (Assuming model predicts noise 'eps', which is v in Scaled context)
        v = model(unet_input, sigma)
        # x0_pred = xt - sig1 * v # Predict x0
        noise_pred = v # Model output is noise prediction v

        # Denoise step (Euler method simplified)
        xt = xt + noise_pred * (sig2 - sig1)

    # Final step to get x0
    # sigma_last = sigmas[-1] * torch.ones(batchsize, device=device)
    # v_last = model(torch.cat([xt, siren_cond], dim=1), sigma_last)
    # x0 = xt - sigma_last * v_last
    # Use the noise prediction from the second to last step? Let's refine.
    # Standard Euler step for the last interval:
    x0 = xt # xt is the state after the last step update using sigmas[-2] -> sigmas[-1]

    return x0

# --- End Conditional Sampling Function ---

def main(args):
    accelerator = Accelerator()
    device = accelerator.device

    # --- Load Pre-trained SIREN Model ---
    print(f"Loading SIREN model from: {args.siren_checkpoint_path}")
    siren_config = dict(
        ndim=2, num_outs=1, grid_size=args.image_size,
        features=tuple(map(int, args.siren_features.split(','))),
        use_scales=args.siren_use_scales, use_shifts=args.siren_use_shifts,
        w0=args.siren_w0, w=args.siren_w, grid_range=(-1, 1)
    )
    siren_model = AutoDecoder(
        latent_dim=args.latent_dim, num_embeddings=10, siren_config=siren_config
    )
    try:
        siren_state_dict = torch.load(args.siren_checkpoint_path, map_location='cpu')
        siren_model.load_state_dict(siren_state_dict)
    except Exception as e:
        print(f"Error loading SIREN checkpoint: {e}")
        return
    siren_model.eval().requires_grad_(False).to(device)
    print("SIREN model loaded successfully.")

    # --- Load Pre-trained UNet Model ---
    print(f"Loading UNet model from: {args.unet_checkpoint_path}")
    unet_model = Scaled(Unet)(
        in_dim=args.image_size, in_ch=2, out_ch=1, # in_ch=2 for conditioning
        ch=args.model_channels,
        ch_mult=tuple(map(int, args.channel_mult.split(','))),
        num_res_blocks=args.num_res_blocks,
        attn_resolutions=tuple(map(int, args.attn_resolutions.split(','))),
        dropout=args.dropout, cond_embed=None, use_coordconv=False
    )
    try:
        unet_state_dict = torch.load(args.unet_checkpoint_path, map_location='cpu')
        unet_model.load_state_dict(unet_state_dict)
    except Exception as e:
        print(f"Error loading UNet checkpoint: {e}")
        return
    unet_model.eval().to(device) # No gradients needed for sampling
    print("UNet model loaded successfully.")

    # --- Setup Diffusion Schedule ---
    schedule = ScheduleLogLinear(sigma_min=args.sigma_min, sigma_max=args.sigma_max, N=args.num_steps)

    # --- Generate Samples ---
    print("Generating samples...")
    num_samples_per_digit = args.num_samples_per_digit
    sample_labels = torch.arange(10, device=device).repeat_interleave(num_samples_per_digit)
    total_samples_to_gen = len(sample_labels)
    sample_bs = args.sample_batch_size

    generated_samples = []
    for i in range(0, total_samples_to_gen, sample_bs):
        batch_labels = sample_labels[i:i+sample_bs]
        current_batch_size = len(batch_labels) # Handle last batch potentially being smaller
        print(f"Generating batch {i//sample_bs + 1}/{(total_samples_to_gen + sample_bs - 1)//sample_bs}...")

        samples_batch = conditional_samples(
            unet_model,
            schedule,
            siren_model,
            batch_labels,
            current_batch_size,
            accelerator # Pass accelerator for device handling
        )
        generated_samples.append(samples_batch.cpu()) # Move to CPU

    generated_samples = torch.cat(generated_samples, dim=0)

    # --- Save Samples ---
    img_grid = make_grid(img_normalize(generated_samples), nrow=num_samples_per_digit)
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    save_image(img_grid, args.output_path)
    print(f"Samples saved successfully to {args.output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample from UNet Diffusion Model with SIREN Conditioning')

    # --- Paths ---
    parser.add_argument('--unet_checkpoint_path', type=str, required=True, help='Path to the trained UNet checkpoint')
    parser.add_argument('--siren_checkpoint_path', type=str, default='models/siren_autoencoder.pth', help='Path to the pre-trained SIREN checkpoint')
    parser.add_argument('--output_path', type=str, default='samples/unet_siren_cond_samples.png', help='Path to save the output sample grid')

    # --- Sampling Params ---
    parser.add_argument('--sample_batch_size', type=int, default=64, help='Batch size for generating samples')
    parser.add_argument('--num_samples_per_digit', type=int, default=8, help='Number of samples to generate per digit (0-9)')

    # --- Diffusion Params (Must match training) ---
    parser.add_argument('--num_steps', type=int, default=800, help='Number of diffusion steps (schedule N)')
    parser.add_argument('--sigma_min', type=float, default=0.01, help='Minimum noise sigma')
    parser.add_argument('--sigma_max', type=float, default=20.0, help='Maximum noise sigma')

    # --- UNet Model Params (Must match training checkpoint) ---
    parser.add_argument('--image_size', type=int, default=28, help='Input image size')
    parser.add_argument('--model_channels', type=int, default=64, help='Base channels for UNet')
    parser.add_argument('--channel_mult', type=str, default='1,1,2', help='Channel multipliers for UNet layers')
    parser.add_argument('--num_res_blocks', type=int, default=2, help='Number of ResNet blocks per resolution')
    parser.add_argument('--attn_resolutions', type=str, default='14', help='Resolutions for attention layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    # --- SIREN Model Params (Must match SIREN checkpoint) ---
    parser.add_argument('--latent_dim', type=int, default=64, help='SIREN latent dimension')
    parser.add_argument('--siren_features', type=str, default='64,64,64', help='SIREN features per layer')
    parser.add_argument('--siren_w0', type=float, default=30.0, help='SIREN w0 frequency')
    parser.add_argument('--siren_w', type=float, default=1.0, help='SIREN w frequency')
    parser.add_argument('--siren_use_scales', action='store_true', help='Use scale modulation in SIREN')
    parser.add_argument('--siren_use_shifts', action='store_true', help='Use shift modulation in SIREN')

    args = parser.parse_args()
    main(args) 