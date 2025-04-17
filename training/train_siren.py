# training/train_siren_autoencoder.py

import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST # Or your custom dataset
from torchvision.transforms import Compose, ToTensor, Lambda # Basic transforms
from torchvision.utils import make_grid, save_image # Added imports
from torch_ema import ExponentialMovingAverage as EMA
import sys
import os
import argparse
from tqdm.auto import tqdm # Use tqdm.auto for better notebook compatibility
import wandb
from functools import partial

# --- Add project root to path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

# --- Imports from smalldiffusion ---
# Assuming siren_model.py is created in src/smalldiffusion/
from siren_model import AutoDecoder 
from smalldiffusion.data import MappedDataset, img_normalize # Assuming img_normalize is helpful

# --- Basic Image Transform ---
# Replace with your specific transforms if needed (e.g., tiling from mnist_unet.py)
# For simplicity now, just resize and normalize
img_transform = Compose([
    ToTensor(),
    Lambda(lambda x: (x * 2) - 1) # Normalize to [-1, 1] if needed by SIREN output range
])

# --- Main Training Function ---
def main(args):
    # --- Setup ---
    accelerator = Accelerator(log_with="wandb" if args.use_wandb else None)
    
    if accelerator.is_main_process and args.use_wandb:
        wandb_config = vars(args) # Log all args
        accelerator.init_trackers(
            project_name=args.wandb_project, 
            config=wandb_config,
            init_kwargs={"wandb": {"name": args.wandb_run_name}}
        )
        
    # Ensure output directory exists
    save_dir = os.path.join(project_root, args.model_save_dir)
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, args.checkpoint_name)

    # --- Data Loading ---
    print("Loading dataset...")
    dataset_path = os.path.join(project_root, "datasets") # Consistent dataset location
    
    # --- Choose Dataset --- # <<<< MODIFY AS NEEDED >>>>
    # --- Single Image Overfitting Test ---
    print("Setting up for single image overfitting test...")
    full_dataset = MNIST(dataset_path, train=True, download=True, transform=img_transform) # Use training set
    # single_image_dataset = Subset(full_dataset, [0]) # Select only the first image
    subset_10_images = Subset(full_dataset, range(10)) # Select the first 10 images
    # num_embeddings = 1 # Only one embedding needed
    num_embeddings = 10 # One embedding per image
    # print(f"Using 1 image from MNIST training dataset (index 0).")
    print(f"Using first 10 images from MNIST training dataset (indices 0-9).")
         
    # We need indices for the AutoDecoder embedding layer
    # Create a dataset that returns (index 0, image 0)
    # Create a dataset that returns (index i, image i) for i in 0..9
    class IndexedDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset):
            self.base_dataset = base_dataset
        def __len__(self):
            # return 1 # Only one item in the dataset for overfitting
            return 10 # 10 items for the 10 images
        def __getitem__(self, idx):
            # Get image 0, discard original label
            # Always return index 0 and image 0
            # return torch.tensor(0, dtype=torch.long), img 
            # Get image corresponding to idx, discard original label
            img, _ = self.base_dataset[idx] 
            # Return index idx and image idx
            return torch.tensor(idx, dtype=torch.long), img 

    # dataset = IndexedDataset(single_image_dataset)
    dataset = IndexedDataset(subset_10_images)
    # loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True) # Batch size 1, no shuffle
    # Use batch size 10 and shuffle
    loader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=4, pin_memory=True) 
    # print(f"Dataset size: {len(dataset)}, Loader size: {len(loader)} batches of 1")
    print(f"Dataset size: {len(dataset)}, Loader size: {len(loader)} batches of 10")

    # --- Model Initialization ---
    print("Initializing SIREN AutoDecoder model...")
    # Define the SIREN config dictionary (adjust parameters as needed)
    siren_config = dict(
        ndim=2, # Assuming 2D images
        # num_outs=1 if full_dataset.data.ndim == 3 else full_dataset.data.shape[-1], # Adjust for channels (1 for MNIST grayscale)
        # grid_size=full_dataset.data.shape[-1], # Assuming square images, e.g., 28 for MNIST
        num_outs=1, # MNIST is grayscale
        grid_size=28, # MNIST is 28x28
        features=tuple(map(int, args.siren_features.split(','))), # e.g., (64, 64, 64)
        use_scales=args.siren_use_scales,
        use_shifts=args.siren_use_shifts,
        w0=args.siren_w0,
        w=args.siren_w,
        grid_range=(-1, 1) # Assuming data normalized to [-1, 1]
    )
    
    model = AutoDecoder(
        latent_dim=args.latent_dim, 
        # num_embeddings=1, # Explicitly set to 1
        num_embeddings=num_embeddings, # Set to 10
        siren_config=siren_config
    )

    # --- Optimizer, Loss, EMA ---
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    reconstruction_loss_fn = nn.MSELoss() # Or nn.L1Loss()
    ema = EMA(model.parameters(), decay=args.ema_decay)
    ema.to(accelerator.device) # Ensure EMA is on the correct device

    # --- Accelerator Prepare ---
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    # --- Training Loop ---
    print("Starting SIREN AutoDecoder training...")
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss_sum = 0.0
        num_batches_epoch = 0

        desc = f"Epoch {epoch+1}/{args.epochs}"
        pbar = tqdm(loader, desc=desc, disable=not accelerator.is_main_process)

        for batch in pbar:
            indices, target_images = batch # Expecting (index, image) from loader
            
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                
                # Forward pass
                reconstructed_images = model(indices)
                
                loss = reconstruction_loss_fn(reconstructed_images, target_images)
                
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
            print(f"Epoch {epoch+1} Average Reconstruction Loss: {avg_epoch_loss:.6f}")
            if args.use_wandb:
                accelerator.log({"train_loss_epoch": avg_epoch_loss, "epoch": epoch + 1}, step=global_step)

            # Optional: Visualize reconstructions (e.g., save first batch)
            if args.save_reconstruction_epoch > 0 and (epoch + 1) % args.save_reconstruction_epoch == 0:
                 print("Saving reconstruction examples...")
                 model.eval()
                 with ema.average_parameters(): # Use EMA weights for viz
                      with torch.no_grad():
                           val_indices, val_images = next(iter(loader)) # Get a validation batch
                           # Ensure indices and images stay on correct device after prepare
                           val_indices = val_indices.to(accelerator.device)
                           val_images = val_images.to(accelerator.device)
                           
                           recons = model(val_indices)
                           recons = accelerator.gather(recons).cpu()
                           val_images = accelerator.gather(val_images).cpu()
                           
                           # Create grid (assuming images are [-1, 1])
                           grid_orig = make_grid(img_normalize(val_images[:16]), nrow=4) 
                           grid_recon = make_grid(img_normalize(recons[:16]), nrow=4)
                           grid_combined = torch.cat((grid_orig, grid_recon), dim=1) # Orig on top, recon below
                           
                           recon_dir = os.path.join(project_root, "reconstructions")
                           os.makedirs(recon_dir, exist_ok=True)
                           recon_path = os.path.join(recon_dir, f"siren_recon_epoch_{epoch+1}.png")
                           save_image(grid_combined, recon_path)
                           print(f"Reconstructions saved to {recon_path}")
                           if args.use_wandb:
                               wandb.log({"reconstructions": wandb.Image(recon_path)}, step=global_step)
                 model.train() # Back to train mode

            # Periodic Checkpoint Saving
            if args.save_every_epochs > 0 and (epoch + 1) % args.save_every_epochs == 0:
                print(f"Saving periodic checkpoint for epoch {epoch+1}...")
                save_path_periodic = checkpoint_path.replace(".pth", f"_epoch_{epoch+1}.pth")
                unwrapped_model = accelerator.unwrap_model(model)
                with ema.average_parameters(): # Save EMA weights
                     torch.save(unwrapped_model.state_dict(), save_path_periodic)
                print(f"Periodic checkpoint saved to {save_path_periodic}")

    # --- Final Saving ---
    if accelerator.is_main_process:
        print(f"Saving final SIREN AutoDecoder model to {checkpoint_path}...")
        unwrapped_model = accelerator.unwrap_model(model)
        with ema.average_parameters(): # Save final EMA weights
             torch.save(unwrapped_model.state_dict(), checkpoint_path)
        print("Final model saved.")

    if accelerator.is_main_process and args.use_wandb:
        wandb.finish()
    print("Training complete.")

# --- Argument Parser ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SIREN AutoEncoder Model')
    # Training Params
    # parser.add_argument('--train_batch_size', type=int, default=64, help='Training batch size') # Default adjusted below
    parser.add_argument('--train_batch_size', type=int, default=10, help='Training batch size (defaults to 10 for 10 images)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay rate')
    # Model Params
    parser.add_argument('--latent_dim', type=int, default=64, help='Dimension of the latent code z and embedding')
    parser.add_argument('--siren_features', type=str, default='64,64,64', help='Comma-separated list of features per SIREN layer')
    parser.add_argument('--siren_w0', type=float, default=30.0, help='w0 frequency for first SIREN layer')
    parser.add_argument('--siren_w', type=float, default=1.0, help='w frequency for subsequent SIREN layers')
    parser.add_argument('--siren_use_scales', action='store_true', help='Use scale modulation in SIREN')
    parser.add_argument('--siren_use_shifts', action='store_true', help='Use shift modulation in SIREN (default is scales only)')
    # Paths and Saving
    parser.add_argument('--model_save_dir', type=str, default='models', help='Directory to save model checkpoints')
    parser.add_argument('--checkpoint_name', type=str, default='siren_autoencoder.pth', help='Filename for the saved checkpoint')
    parser.add_argument('--save_every_epochs', type=int, default=10, help='Save checkpoint every N epochs (0 disables periodic saving)')
    parser.add_argument('--save_reconstruction_epoch', type=int, default=5, help='Save reconstruction image grid every N epochs (0 disables)')
    # Wandb
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='siren-autoencoder-train', help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default='siren-ae-run', help='Wandb run name')
    # Dataset (Placeholder - adjust as needed)
    # parser.add_argument('--dataset_path', type=str, default='datasets', help='Path to datasets directory')
    # parser.add_argument('--dataset_name', type=str, default='mnist', help='Name of dataset to use')

    args = parser.parse_args()
    
    # Basic validation for modulation
    if not args.siren_use_scales and not args.siren_use_shifts:
        print("Warning: Neither scale nor shift modulation is enabled for SIREN. Using scale modulation by default.")
        args.siren_use_scales = True # Default to scales if none selected

    main(args)