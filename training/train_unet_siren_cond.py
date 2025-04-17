import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchvision import transforms as tf
from torchvision.datasets import MNIST # Changed from FashionMNIST
from torchvision.utils import make_grid, save_image
from torch_ema import ExponentialMovingAverage as EMA
import sys
import os
import argparse # Added argparse
import tqdm # Use tqdm if available, otherwise keep default Accelerate progress
import wandb # Added wandb
import torch.optim as optim # Added optimizer import
import torch.nn as nn # Added for loss function

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, 'src') # Define src_path
sys.path.insert(0, project_root) # Add project root for smalldiffusion package
sys.path.insert(1, src_path)    # Add src path for siren_model

# Imports from smalldiffusion
# from src.smalldiffusion.diffusion import ScheduleLogLinear, samples, training_loop # Remove training_loop, samples
from smalldiffusion.diffusion import ScheduleLogLinear # Keep Schedule
from smalldiffusion.data import MappedDataset, img_train_transform, img_normalize
from smalldiffusion.model import Scaled # Import Scaled from model.py
from smalldiffusion.model_unet import Unet # Import Unet from model_unet.py
# Import SIREN model (now directly importable)
from siren_model import AutoDecoder, SpatialModulationBlock
# Import the new transform from utils - KEEPING THIS for now, might simplify later
from smalldiffusion.utils.data_transforms import img_tile_train_transform
# Import metric calculation utility - REMOVING THIS, focusing on reconstruction loss first
# from smalldiffusion.utils.metrics import calculate_quadrant_mse # Removed src.

# --- Simplified Conditional Sampling Function REMOVED ---
# @torch.no_grad()
# def conditional_samples(...): ...
# --- End Conditional Sampling Function REMOVED ---


def main(args):
    # Setup
    accelerator = Accelerator(log_with="wandb" if args.use_wandb else None)

    siren_model = None
    if not args.disable_siren_cond:
        # --- SIREN Model Loading (Conditional) ---
        print("Loading pre-trained SIREN model for conditioning...")
        # Define SIREN config (must match the trained model) - Adjust if needed
        siren_config = dict(
            ndim=2,
            num_outs=1,
            grid_size=args.image_size, # Use image_size arg
            features=tuple(map(int, args.siren_features.split(','))),
            use_scales=args.siren_use_scales,
            use_shifts=args.siren_use_shifts,
            w0=args.siren_w0,
            w=args.siren_w,
            grid_range=(-1, 1)
        )
        siren_model = AutoDecoder(
            latent_dim=args.latent_dim,
            num_embeddings=10, # Should be 10 for 0-9 digits
            siren_config=siren_config
        )
        try:
            siren_state_dict = torch.load(args.siren_checkpoint_path, map_location='cpu')
            siren_model.load_state_dict(siren_state_dict)
            print(f"SIREN checkpoint loaded successfully from {args.siren_checkpoint_path}")
        except Exception as e:
            print(f"Error loading SIREN checkpoint: {e}")
            print("Cannot proceed with SIREN conditioning enabled without a valid SIREN model. Exiting.")
            return # Exit if SIREN model fails to load

        siren_model.eval() # Set SIREN to evaluation mode
        siren_model.requires_grad_(False) # Freeze SIREN parameters
        siren_model = siren_model.to(accelerator.device) # Move SIREN to device
        # --- End SIREN Model Loading ---
    else:
        print("SIREN conditioning disabled.")

    if accelerator.is_main_process and args.use_wandb:
        cond_suffix = "siren_cond" if not args.disable_siren_cond else "no_cond"
        siren_chkpt_name = args.siren_checkpoint_path.split('/')[-1] if not args.disable_siren_cond else "NA"
        run_name = f"unet-{cond_suffix}-{siren_chkpt_name}"
        wandb_config = vars(args)
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=wandb_config,
            init_kwargs={"wandb": {"name": run_name}}
        )

    # Ensure output directories exist
    save_dir = os.path.join(project_root, args.model_save_dir)
    # img_dir = os.path.join(project_root, args.sample_save_dir) # REMOVED - Arg doesn't exist anymore
    os.makedirs(save_dir, exist_ok=True)
    # os.makedirs(img_dir, exist_ok=True) # REMOVED

    checkpoint_path = os.path.join(save_dir, args.checkpoint_name)
    # sample_path = os.path.join(img_dir, f"samples_{args.checkpoint_name.replace('.pth', '.png')}") # REMOVED - Not saving samples here

    print("Loading MNIST dataset...")
    dataset_path = os.path.join(project_root, "datasets")
    # Use standard MNIST transform for now, tiling might not be necessary
    img_transform = tf.Compose([tf.ToTensor(), tf.Lambda(lambda x: (x * 2) - 1)])
    dataset = MappedDataset(MNIST(dataset_path, train=True, download=True,
                                  transform=img_transform),
                            lambda x: x) # Keep both image and label
    loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    print(f"Dataset size: {len(dataset)}, Loader size: {len(loader)} batches of {args.train_batch_size}")

    print("Initializing UNet model and schedule...")
    schedule = ScheduleLogLinear(sigma_min=args.sigma_min, sigma_max=args.sigma_max, N=args.num_steps)
    # --- Instantiate UNet with conditional in_ch ---
    unet_in_channels = 1 if args.disable_siren_cond else 2
    print(f"Initializing UNet model with {unet_in_channels} input channels...")
    model = Scaled(Unet)(
        in_dim=args.image_size, # Should be 28 for MNIST
        in_ch=unet_in_channels,
        out_ch=1,             # Output channels (noise prediction)
        ch=args.model_channels,
        ch_mult=tuple(map(int, args.channel_mult.split(','))),
        num_res_blocks=args.num_res_blocks,
        attn_resolutions=tuple(map(int, args.attn_resolutions.split(','))),
        dropout=args.dropout,
        cond_embed=None, # Explicitly set to None
        use_coordconv=False # Keep coordconv off for simplicity first
    )

    # --- Load Checkpoint for UNet if specified ---
    if args.load_checkpoint:
        print(f"Loading UNet checkpoint: {args.load_checkpoint}")
        try:
            state_dict = torch.load(args.load_checkpoint, map_location='cpu')
            # Basic check for input layer shape mismatch
            current_in_channels = model.module.conv_in.conv.in_channels
            ckpt_in_channels = state_dict.get('module.conv_in.conv.weight', state_dict.get('conv_in.conv.weight')).shape[1]

            if ckpt_in_channels != current_in_channels:
                 print(f"Warning: Checkpoint input channels ({ckpt_in_channels}) mismatch model ({current_in_channels}). Attempting partial load.")
                 # Remove incompatible conv_in weights/biases
                 state_dict.pop('module.conv_in.conv.weight', None)
                 state_dict.pop('conv_in.conv.weight', None)
                 state_dict.pop('module.conv_in.conv.bias', None)
                 state_dict.pop('conv_in.conv.bias', None)
                 model.load_state_dict(state_dict, strict=False)
            else:
                 model.load_state_dict(state_dict, strict=True)
            print(f"UNet Checkpoint loaded successfully.")
        except Exception as e:
            print(f"Error loading UNet checkpoint: {e}. Proceeding from scratch.")
    else:
        print("No UNet checkpoint specified, starting from scratch.")
    # --- End Load Checkpoint ---

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    ema = EMA(model.parameters(), decay=args.ema_decay)
    loss_fn = nn.MSELoss()

    # Prepare model, optimizer, loader
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    ema.to(accelerator.device) # Ensure EMA is on correct device after prepare

    print("Starting training...")
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss_sum = 0.0
        num_batches_epoch = 0

        desc = f"Epoch {epoch+1}/{args.epochs}"
        pbar = tqdm.tqdm(loader, desc=desc, disable=not accelerator.is_main_process)

        for batch in pbar:
            images, labels = batch
            # Ensure labels are long type and on correct device
            labels = labels.to(accelerator.device).long()

            with accelerator.accumulate(model):
                optimizer.zero_grad()

                # Sample sigmas (timesteps)
                sigmas_batch = schedule.sample_batch(images).to(accelerator.device) # CORRECT: Use training schedule sampling, shape [B]
                # Reshape for broadcasting during noisy image creation
                # while len(sigmas.shape) < len(images.shape):
                #      sigmas = sigmas.unsqueeze(-1)
                sigmas_broadcast = sigmas_batch.reshape(images.shape[0], 1, 1, 1) # Shape [B, 1, 1, 1]

                # Sample noise
                noise = torch.randn_like(images)
                # Create noisy image
                x_noisy = images + noise * sigmas_broadcast

                # --- Generate SIREN Condition (Conditional) ---
                siren_cond = None
                if not args.disable_siren_cond:
                    with torch.no_grad():
                        siren_cond = siren_model(labels)
                # --- End SIREN Condition ---

                # --- Prepare UNet Input (Conditional) ---
                if args.disable_siren_cond:
                    unet_input = x_noisy
                else:
                    # Ensure siren_cond is 4D [B, 1, H, W] before cat - should be already, but defensive check
                    if siren_cond is None or siren_cond.dim() != 4:
                        raise ValueError(f"SIREN condition has unexpected shape or is None: {siren_cond.shape if siren_cond is not None else 'None'}")
                    if x_noisy.dim() != 4:
                        raise ValueError(f"Noisy image has unexpected shape: {x_noisy.shape}")
                    unet_input = torch.cat([x_noisy, siren_cond], dim=1)
                # --- End UNet Input ---

                # Predict noise using UNet - Pass the original batch sigma [B]
                predicted_noise = model(unet_input, sigmas_batch)

                # Calculate loss
                loss = loss_fn(predicted_noise, noise)

                accelerator.backward(loss)
                optimizer.step()
                ema.update()

                # Logging
                loss_item = accelerator.gather(loss).mean().item()
                epoch_loss_sum += loss_item
                num_batches_epoch += 1
                if accelerator.is_main_process:
                    pbar.set_postfix(Loss=loss_item)
                    if args.use_wandb:
                        accelerator.log({"train_loss_step": loss_item}, step=global_step)
                global_step += 1

        # --- End of Epoch ---
        avg_epoch_loss = epoch_loss_sum / num_batches_epoch if num_batches_epoch > 0 else 0.0
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch+1} Average Loss: {avg_epoch_loss:.6f}")
            if args.use_wandb:
                accelerator.log({"train_loss_epoch": avg_epoch_loss, "epoch": epoch + 1}, step=global_step)

            # Periodic Checkpoint Saving (UNet)
            if args.save_every_epochs > 0 and (epoch + 1) % args.save_every_epochs == 0:
                print(f"Saving periodic UNet checkpoint for epoch {epoch+1}...")
                save_path_periodic = checkpoint_path.replace(".pth", f"_epoch_{epoch+1}.pth")
                unwrapped_model = accelerator.unwrap_model(model)
                with ema.average_parameters(): # Save EMA weights
                     torch.save(unwrapped_model.state_dict(), save_path_periodic)
                print(f"Periodic UNet checkpoint saved to {save_path_periodic}")

    # --- Final Saving ---
    if accelerator.is_main_process:
        print(f"Saving final UNet model to {checkpoint_path}...")
        unwrapped_model = accelerator.unwrap_model(model)
        with ema.average_parameters(): # Save final EMA weights
             torch.save(unwrapped_model.state_dict(), checkpoint_path)
        print("Final UNet model saved.")

    if accelerator.is_main_process and args.use_wandb:
        wandb.finish()
    print("Training complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train UNet Diffusion Model with SIREN Conditioning on MNIST')

    # --- General Training Params ---
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--train_batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay rate')

    # --- Diffusion Params ---
    parser.add_argument('--num_steps', type=int, default=800, help='Number of diffusion steps (schedule N)')
    parser.add_argument('--sigma_min', type=float, default=0.01, help='Minimum noise sigma')
    parser.add_argument('--sigma_max', type=float, default=20.0, help='Maximum noise sigma')

    # --- UNet Model Params ---
    parser.add_argument('--image_size', type=int, default=28, help='Input image size')
    parser.add_argument('--model_channels', type=int, default=64, help='Base channels for UNet')
    parser.add_argument('--channel_mult', type=str, default='1,1,2', help='Channel multipliers for UNet layers (comma-separated)')
    parser.add_argument('--num_res_blocks', type=int, default=2, help='Number of ResNet blocks per resolution')
    parser.add_argument('--attn_resolutions', type=str, default='14', help='Resolutions for attention layers (comma-separated)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    # parser.add_argument('--coordconv_mode', type=str, default='none', choices=['none', 'minimal', 'layer1_layer2', 'full', 'maximal'], help='CoordConv usage mode in UNet')
    parser.add_argument('--disable_siren_cond', action='store_true', help='Disable SIREN conditioning and run standard UNet')

    # --- SIREN Model Params (Only relevant if conditioning is enabled) ---
    parser.add_argument('--siren_checkpoint_path', type=str, default='models/siren_autoencoder.pth', help='Path to the pre-trained SIREN AutoDecoder checkpoint')
    parser.add_argument('--latent_dim', type=int, default=64, help='SIREN latent dimension (must match checkpoint)')
    parser.add_argument('--siren_features', type=str, default='64,64,64', help='SIREN features per layer (must match checkpoint)')
    parser.add_argument('--siren_w0', type=float, default=30.0, help='SIREN w0 frequency (must match checkpoint)')
    parser.add_argument('--siren_w', type=float, default=1.0, help='SIREN w frequency (must match checkpoint)')
    parser.add_argument('--siren_use_scales', action='store_true', help='Use scale modulation in SIREN (must match checkpoint)')
    parser.add_argument('--siren_use_shifts', action='store_true', help='Use shift modulation in SIREN (must match checkpoint)')

    # --- Saving/Loading Params ---
    parser.add_argument('--model_save_dir', type=str, default='models', help='Directory to save UNet model checkpoints')
    parser.add_argument('--checkpoint_name', type=str, default='mnist_unet_siren_cond.pth', help='Filename for the saved UNet checkpoint')
    parser.add_argument('--save_every_epochs', type=int, default=20, help='Save UNet checkpoint every N epochs (0 disables)')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Path to UNet checkpoint to load and continue training')

    # --- Sampling/Logging Params ---
    # parser.add_argument('--sample_save_dir', type=str, default='samples', help='Directory to save generated samples') # Arg unused now
    # parser.add_argument('--sample_batch_size', type=int, default=64, help='Batch size for generating samples') # Arg unused now
    # parser.add_argument('--save_samples_epoch', type=int, default=10, help='Save sample image grid every N epochs (0 disables)') # Arg unused now
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='smalldiffusion-siren-cond', help='Wandb project name')

    args = parser.parse_args()

    # Validate SIREN modulation args if checkpoint exists (optional but good practice)
    # Could add logic here to infer siren settings from checkpoint name if needed

    main(args)