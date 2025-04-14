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

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Imports from smalldiffusion
from src.smalldiffusion.diffusion import ScheduleLogLinear, samples, training_loop
from src.smalldiffusion.data import MappedDataset, img_train_transform, img_normalize
from src.smalldiffusion.model import Scaled # Import Scaled from model.py
from src.smalldiffusion.model_unet import Unet # Import Unet from model_unet.py
# Import the new transform from utils
from src.smalldiffusion.utils.data_transforms import img_tile_train_transform
# Import metric calculation utility
from src.smalldiffusion.utils.metrics import calculate_quadrant_mse

def main(args):
    # Setup
    # Determine CoordConv configuration based on mode
    use_coordconv_in_unet = args.coordconv_mode != 'none'
    # Note: ResnetBlock modification needs to be done in model_unet.py based on the mode.
    # This script will just pass the flag, assuming the Unet/ResnetBlock interpret it correctly
    # OR we modify Unet to accept the mode string directly.
    # For simplicity now, let's assume the loaded model_unet.py corresponds to the desired mode.
    # We'll control which code version of model_unet.py we use implicitly.
    # This is NOT ideal, but avoids complex conditional logic in model loading/init here.
    # Ideally, Unet __init__ would take the mode. 

    # Modify run name and suffix based on mode
    mode_suffix_map = {
        'none': "",
        'minimal': "_minimal_coords", # Layer1 only
        'layer1_layer2': "_layer1_layer2_coords", # Layer1+Layer2, no shortcut
        'full': "_full_coords", # Layer1+Layer2+Shortcut + conv_in
        'maximal': "_maximal_coords" # Full + Downsample/Upsample convs
    }
    coordconv_suffix = mode_suffix_map.get(args.coordconv_mode, "_unknown_coords")
    run_name = f"unet-{args.coordconv_mode}-tiled_fine_tune"
    
    wandb.init(project="smalldiffusion-tiled-comparison", config=args, name=run_name)
    
    # Ensure accelerator uses the correct visible devices (set via env var)
    accelerator = Accelerator()
    
    # Ensure output directories exist
    save_dir = os.path.join(project_root, args.model_save_dir)
    img_dir = os.path.join(project_root, args.sample_save_dir)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    # Adjust filenames based on coordconv_mode
    dataset_suffix = "_tiled"
    checkpoint_path = os.path.join(save_dir, f"mnist_unet{coordconv_suffix}{dataset_suffix}_checkpoint.pth")
    sample_path = os.path.join(img_dir, f"mnist_unet{coordconv_suffix}{dataset_suffix}_samples.png")

    print("Loading MNIST dataset for Tiled Fine-tuning...")
    dataset_path = os.path.join(project_root, "datasets") # Store datasets in root/datasets
    # Use the new transform pipeline
    dataset = MappedDataset(MNIST(dataset_path, train=True, download=True,
                                  transform=img_tile_train_transform), # Use new transform
                            lambda x: x[0]) # Get only the image, discard label
    loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    print(f"Dataset size: {len(dataset)}, Loader size: {len(loader)} batches of {args.train_batch_size}")

    print("Initializing model and schedule...")
    # Pass the boolean flag based on whether mode is 'none'
    print(f"CoordConv Mode: {args.coordconv_mode}") 
    schedule = ScheduleLogLinear(sigma_min=0.01, sigma_max=20, N=800)
    model = Scaled(Unet)(in_dim=28, in_ch=1, out_ch=1, ch=64, ch_mult=(1, 1, 2), 
                           attn_resolutions=(14,), use_coordconv=use_coordconv_in_unet)

    # --- Load Checkpoint if specified --- 
    # Important: Assumes the loaded checkpoint matches the architecture specified by coordconv_mode
    # E.g., loading a standard checkpoint when mode is 'full' might fail or need strict=False
    if args.load_checkpoint:
        print(f"Loading checkpoint: {args.load_checkpoint} for mode {args.coordconv_mode}")
        try:
            # Use strict=True when fine-tuning same architecture
            strict_loading = (args.coordconv_mode == 'none') == ("coordconv" not in args.load_checkpoint.lower())
            state_dict = torch.load(args.load_checkpoint, map_location='cpu')
            model.load_state_dict(state_dict, strict=strict_loading)
            print(f"Checkpoint loaded successfully (strict={strict_loading}).")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Proceeding with randomly initialized model.")
    else:
        print("No checkpoint specified, starting from scratch.")
    # --- End Load Checkpoint --- 

    # Initialize optimizer and EMA directly - Reverted this part, trainer handles it.
    # Optimizer and EMA setup is now inside the training_loop
    # We still need EMA instance here for saving and evaluation
    ema = EMA(model.parameters(), decay=0.999) 
    ema.to(accelerator.device)

    # Prepare model and loader only 
    model, loader = accelerator.prepare(model, loader)

    print("Starting training...")
    # Train using the diffusion.training_loop again
    trainer = training_loop(
        loader,
        model,
        schedule,
        accelerator=accelerator,
        epochs=args.epochs,
        lr=args.lr
    )

    # Use tqdm for progress bar if installed
    desc = f"Training MNIST Unet {args.coordconv_mode}{dataset_suffix}"
    pbar = None
    try:
        pbar = tqdm.tqdm(trainer, total=args.epochs * len(loader), desc=desc, disable=not accelerator.is_main_process)
    except ImportError:
        if accelerator.is_main_process:
            print("tqdm not found, using standard progress.")
        pbar = trainer

    last_saved_epoch = -1 
    epoch_loss_sum = 0.0
    num_batches_epoch = 0
    current_epoch_tracked = -1 # Track epoch changes

    for i, stats in enumerate(pbar):
        ema.update() # Update EMA after each training step/yield

        # Gather loss for logging
        loss_item = accelerator.gather(stats.loss).mean().item()
        epoch_loss_sum += loss_item
        num_batches_epoch += 1
        
        # Update progress bar
        if isinstance(pbar, tqdm.tqdm):
             pbar.set_postfix(Epoch=getattr(stats, 'current_epoch', -1)+1, Loss=loss_item)
        elif accelerator.is_main_process and (i + 1) % 50 == 0: 
             print(f"Step {i+1}, Loss={loss_item:.5f}")

        # Check if epoch changed (using stats.current_epoch added to training_loop)
        if hasattr(stats, 'current_epoch') and stats.current_epoch != current_epoch_tracked:
            # --- End of Epoch Actions --- 
            if current_epoch_tracked != -1: # Don't run on first iteration (before epoch 0 ends)
                # Calculate average loss for the completed epoch
                avg_epoch_loss = epoch_loss_sum / num_batches_epoch
                if accelerator.is_main_process:
                    print(f"\nEpoch {current_epoch_tracked + 1}/{args.epochs} completed. Average Loss: {avg_epoch_loss:.5f}")
                    
                    # --- Wandb Logging & Quadrant MSE Calculation --- 
                    print(f"Epoch {current_epoch_tracked + 1}: Generating samples for evaluation...")
                    model.eval() # Set model to eval for sampling
                    eval_samples = []
                    eval_batch_size = args.sample_batch_size # Use sample batch size for eval
                    num_eval_batches = max(1, 64 // eval_batch_size) # Generate ~64 samples
                    
                    with ema.average_parameters(): # Use EMA weights
                        unwrapped_model_eval = accelerator.unwrap_model(model)
                        for _ in range(num_eval_batches):
                            *_, x0_eval = samples(
                                unwrapped_model_eval, 
                                schedule.sample_sigmas(20), 
                                gam=1.6, 
                                batchsize=eval_batch_size, 
                                accelerator=accelerator
                            )
                            # Gather and move to CPU *inside* the loop to avoid large GPU mem usage
                            eval_samples.append(accelerator.gather(x0_eval).cpu())
                    
                    eval_samples_cat = torch.cat(eval_samples, dim=0)[:64] # Concatenate and limit to 64
                    quadrant_mse = calculate_quadrant_mse(eval_samples_cat)
                    print(f"Epoch {current_epoch_tracked + 1}: Quadrant MSE = {quadrant_mse:.6f}")
                    
                    # Log metrics to wandb
                    wandb.log({
                        "epoch": current_epoch_tracked + 1, 
                        "train_loss_epoch_avg": avg_epoch_loss, 
                        "quadrant_mse_ema": quadrant_mse
                    })
                    model.train() # Set model back to train mode
                    # --- End Wandb Logging --- 

                    # --- Periodic Checkpoint Saving --- 
                    current_epoch_1_based = current_epoch_tracked + 1
                    if (args.save_every_epochs > 0 and 
                        current_epoch_1_based % args.save_every_epochs == 0 and 
                        current_epoch_tracked > last_saved_epoch):
                        
                        save_path_periodic = checkpoint_path.replace(".pth", f"_epoch_{current_epoch_1_based}.pth")
                        print(f"Saving periodic checkpoint to {save_path_periodic}...")
                        with ema.average_parameters(): 
                            unwrapped_model_periodic = accelerator.unwrap_model(model)
                            torch.save(unwrapped_model_periodic.state_dict(), save_path_periodic)
                        print("Periodic checkpoint saved.")
                        last_saved_epoch = current_epoch_tracked # Update last saved epoch (0-based)
                    # --- End Periodic Saving --- 
            
            # Reset for next epoch
            current_epoch_tracked = stats.current_epoch
            epoch_loss_sum = 0.0
            num_batches_epoch = 0
            # --- End End of Epoch Actions ---

    # Final cleanup/saving after the loop finishes
    if isinstance(pbar, tqdm.tqdm):
        pbar.close()
    print("Training complete.")

    # Final Evaluation Log (for the last epoch)
    if accelerator.is_main_process:
        # Calculate final epoch avg loss if needed (might be slightly off if loop exited early)
        if num_batches_epoch > 0:
             avg_epoch_loss = epoch_loss_sum / num_batches_epoch
        else: # Handle case where training was 0 epochs? 
             avg_epoch_loss = 0.0 
        print(f"\nFinal Epoch {current_epoch_tracked + 1}/{args.epochs} completed. Average Loss: {avg_epoch_loss:.5f}")
        # Generate samples & calc MSE for the final epoch state
        model.eval()
        eval_samples = []
        eval_batch_size = args.sample_batch_size
        num_eval_batches = max(1, 64 // eval_batch_size)
        with ema.average_parameters():
            unwrapped_model_eval = accelerator.unwrap_model(model)
            for _ in range(num_eval_batches):
                *_, x0_eval = samples(unwrapped_model_eval, schedule.sample_sigmas(20), gam=1.6, batchsize=eval_batch_size, accelerator=accelerator)
                eval_samples.append(accelerator.gather(x0_eval).cpu())
        eval_samples_cat = torch.cat(eval_samples, dim=0)[:64]
        quadrant_mse = calculate_quadrant_mse(eval_samples_cat)
        print(f"Final Epoch {current_epoch_tracked + 1}: Quadrant MSE = {quadrant_mse:.6f}")
        wandb.log({"epoch": current_epoch_tracked + 1, "train_loss_epoch_avg": avg_epoch_loss, "quadrant_mse_ema": quadrant_mse})

    # Final model saving / sample image saving
    if accelerator.is_main_process:
        print("Generating final samples image...")
        with ema.average_parameters(): # Use EMA weights for final sampling and saving
            unwrapped_model = accelerator.unwrap_model(model)
            *xt, x0 = samples(unwrapped_model, schedule.sample_sigmas(20), gam=1.6,
                              batchsize=args.sample_batch_size, accelerator=accelerator)
            x0_cpu = accelerator.gather(x0).cpu()
            save_image(img_normalize(make_grid(x0_cpu)), sample_path)
            print(f"Samples saved to {sample_path}")

            print(f"Saving final checkpoint to {checkpoint_path}...")
            torch.save(unwrapped_model.state_dict(), checkpoint_path)
            print("Final checkpoint saved.")

    wandb.finish() # Finish wandb run

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train MNIST UNet Diffusion Model')
    parser.add_argument('--train_batch_size', type=int, default=1024, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--sample_batch_size', type=int, default=64, help='Batch size for sampling')
    parser.add_argument('--lr', type=float, default=7e-4, help='Learning rate')
    parser.add_argument('--model_save_dir', type=str, default='models', help='Directory to save model checkpoints')
    parser.add_argument('--sample_save_dir', type=str, default='imgs', help='Directory to save generated samples')
    parser.add_argument('--coordconv_mode', type=str, default='none', 
                        choices=['none', 'minimal', 'layer1_layer2', 'full', 'maximal'], 
                        help='CoordConv configuration mode')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Path to pre-trained model checkpoint for fine-tuning')
    parser.add_argument('--save_every_epochs', type=int, default=0, help='Save checkpoint every N epochs (0 disables periodic saving)')

    args = parser.parse_args()
    main(args)