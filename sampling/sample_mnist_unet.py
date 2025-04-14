#!/usr/bin/env python
import torch
from accelerate import Accelerator
from torchvision.utils import save_image
import sys
import os
import math
import argparse
import tqdm

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Imports from smalldiffusion
from src.smalldiffusion.diffusion import ScheduleLogLinear, samples
from src.smalldiffusion.data import img_normalize
from src.smalldiffusion.model import Scaled
from src.smalldiffusion.model_unet import Unet

def main(args):
    # Seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Setup Accelerator (respects CUDA_VISIBLE_DEVICES)
    accelerator = Accelerator()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving generated samples to: {args.output_dir}")

    # Initialize model
    print(f"Initializing model (CoordConv: {args.use_coordconv})...")
    model = Scaled(Unet)(
        in_dim=28, 
        in_ch=1, 
        out_ch=1, 
        ch=64, 
        ch_mult=(1, 1, 2), 
        attn_resolutions=(14,), 
        use_coordconv=args.use_coordconv
    )

    # Load checkpoint
    print(f"Loading checkpoint: {args.load_checkpoint}")
    try:
        state_dict = torch.load(args.load_checkpoint, map_location='cpu')
        model.load_state_dict(state_dict)
        print("Checkpoint loaded successfully.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        accelerator.print(f"Failed to load checkpoint {args.load_checkpoint}. Exiting.")
        exit(1)

    # Prepare model for Accelerator (moves to correct device)
    model = accelerator.prepare(model)
    model.eval()

    # Initialize schedule
    schedule = ScheduleLogLinear(sigma_min=0.01, sigma_max=20, N=800)
    sampling_sigmas = schedule.sample_sigmas(args.num_steps)

    # Generation loop
    num_generated = 0
    num_batches = math.ceil(args.num_samples / args.batch_size)
    print(f"Generating {args.num_samples} samples in {num_batches} batches (batch size: {args.batch_size})...")

    with torch.no_grad():
        for i in tqdm.tqdm(range(num_batches), desc="Generating Samples", disable=not accelerator.is_main_process):
            current_batch_size = min(args.batch_size, args.num_samples - num_generated)
            if current_batch_size <= 0:
                break

            # Generate samples using the diffusion sampler
            *xt, x0 = samples(
                model,
                sampling_sigmas,
                gam=args.gam,
                batchsize=current_batch_size,
                accelerator=accelerator
            )

            # Gather samples if distributed
            x0_gathered = accelerator.gather(x0)

            # Only save images on the main process
            if accelerator.is_main_process:
                x0_cpu = x0_gathered.cpu()
                # Save individual images
                for j in range(x0_cpu.shape[0]):
                    if num_generated < args.num_samples:
                        img_path = os.path.join(args.output_dir, f"{num_generated:05d}.png")
                        save_image(img_normalize(x0_cpu[j]), img_path)
                        num_generated += 1
                    else:
                        break # Stop if we've generated enough samples
    
    if accelerator.is_main_process:
        print(f"Finished generating {num_generated} samples.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample from a trained MNIST UNet model')
    parser.add_argument('--load_checkpoint', type=str, required=True, help='Path to the model checkpoint (.pth)')
    parser.add_argument('--use_coordconv', action='store_true', help='Specify if the checkpoint uses CoordConv layers')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save generated samples')
    parser.add_argument('--num_samples', type=int, default=64, help='Total number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for generation')
    parser.add_argument('--num_steps', type=int, default=20, help='Number of diffusion sampling steps')
    parser.add_argument('--gam', type=float, default=1.6, help='Gamma parameter for the sampler')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')

    args = parser.parse_args()
    
    # Simple validation
    if not os.path.exists(args.load_checkpoint):
        print(f"Error: Checkpoint file not found: {args.load_checkpoint}")
        exit(1)
        
    main(args) 