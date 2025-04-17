import math
import torch
import torch.nn as nn
# Note: optim, typing.Sequence, cached_property were used in the notebook
# but Sequence and cached_property are needed for the classes below.
from typing import Sequence 
from functools import cached_property

class Sine(nn.Module):
    def __init__(self, freq=1.0):
        super().__init__()
        self.freq = freq

    def forward(self, inputs):
        return torch.sin(self.freq * inputs)

    def __repr__(self):
        return f"Sine(w0={self.freq})"
    
class ModulatedLayer(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        n_dims: int,
        non_linearity = None,
        scale: float = 1.0,
    ):
        super().__init__()
        conv = nn.Conv2d if n_dims == 2 else nn.Conv3d
        self.linear = conv(n_inputs, n_outputs, kernel_size=1, bias=True)
        self.non_linearity = Sine(scale) if non_linearity is None else non_linearity
        self.scale = scale
        self.reset_parameters()

    def reset_parameters(self, start=False):
        n_in = self.linear.in_channels
        if start:
            b = 1 / n_in
        else:
            b = math.sqrt(6 / n_in) / self.scale
        nn.init.uniform_(self.linear.weight, -b, b)
        nn.init.zeros_(self.linear.bias)  # type: ignore

    def forward(self, x, shift_mod=0.0, scale_mod=1.0):
        # The notebook had scale_mod * (self.linear(x) + shift_mod)
        # Let's ensure we apply scale_mod correctly based on typical FiLM-style modulation
        # Usually scale is multiplicative AFTER non-linearity, shift is BEFORE.
        # Re-check Mehta et al. if needed, but let's assume standard FiLM for now:
        # linear_out = self.linear(x) + shift_mod 
        # nonlin_out = self.non_linearity(linear_out)
        # return scale_mod * nonlin_out 
        # ***CORRECTION based on notebook code***: 
        # The notebook code actually does scale * (linear + shift) THEN sine. Let's stick to that.
        return self.non_linearity(scale_mod * (self.linear(x) + shift_mod))
    
class SpatialModulationBlock(nn.Module):
    def __init__(
        self,
        ndim: int,
        num_outs: int,
        grid_size: int,
        features: Sequence[int] = (64, 64, 64, 64),
        latent_dim: int = 16,
        use_shifts: bool = False,
        use_scales: bool = True,
        w0: float = 30.0,
        w: float = 1.0,
        interpolation: str = 'linear', # Note: this interpolation isn't used in the forward pass shown
        grid_range: tuple[float, float] = (-1, 1),
    ):
        super().__init__()
        assert use_shifts or use_scales
        # assert interpolation in ['linear', 'nearest'] # Interpolation not used here

        if ndim == 2:
            conv = nn.Conv2d
            # if interpolation == "linear": interpolation = "bilinear" # Not used
        else:
            conv = nn.Conv3d
            # if interpolation == "linear": interpolation = "trilinear" # Not used

        modulated_layers, layer_inputs = nn.ModuleList(), ndim
        for i in range(len(features)):
            modulated_layers.append(ModulatedLayer(
                n_dims=ndim,
                n_inputs=layer_inputs,
                n_outputs=features[i],
                scale = w0 if i == 0 else w,
            ))
            layer_inputs = features[i]

        output_layer = ModulatedLayer(
            n_inputs=features[-1],
            n_dims=ndim,
            n_outputs=num_outs,
            scale=w,
            non_linearity=nn.Identity(),
        )

        split_sizes = list(features)
        if use_shifts:
            # Using Conv2d/3d assumes z is spatially arranged, maybe MLP is simpler if z is just a vector?
            # The notebook code uses Conv2d with kernel_size=1 which is equivalent to per-pixel MLP
            # Let's keep it as Conv for now, assuming z might be given spatial dimensions later
            # Revert capacity: Conv -> ReLU
            # mod_hidden_dim = sum(features) # Or choose a different intermediate dimension
            self.shift_modulations = nn.Sequential(
                # conv(latent_dim, mod_hidden_dim, kernel_size=1),
                conv(latent_dim, sum(features), kernel_size=1), # Direct mapping
                nn.ReLU() # Mehta used ReLU here
                # conv(mod_hidden_dim, sum(features), kernel_size=1) # Output layer - REMOVED
            )
        else:
            self.shift_modulations = None # Use None instead of lambda for clarity

        if use_scales:
             # Revert capacity: Conv -> ReLU
             # mod_hidden_dim = sum(features) # Match the shift network's intermediate dim
             self.scale_modulations = nn.Sequential(
                # conv(latent_dim, mod_hidden_dim, kernel_size=1),
                conv(latent_dim, sum(features), kernel_size=1), # Direct mapping
                nn.ReLU() # Mehta used ReLU here
                # conv(mod_hidden_dim, sum(features), kernel_size=1) # Output layer - REMOVED
            )
        else:
            self.scale_modulations = None # Use None instead of lambda

        # substrate
        self.ndim = ndim
        self.grid_size = grid_size
        self.grid_range = grid_range
        # latent
        self.latent_dim = latent_dim
        self.use_scales = use_scales
        self.use_shifts = use_shifts
        # layers
        self.modulated_layers = modulated_layers
        self.output_layer = output_layer
        self.split_sizes = split_sizes
        # self.interpolation = interpolation # Not used
        self.modulated_layers[0].reset_parameters(start=True)  # type: ignore

    # Removing reset_parameters method here as it's part of ModulatedLayer
    # def reset_parameters(self): ...

    @cached_property
    def grid(self):
        # Ensure grid is generated on the correct device later
        coords = [torch.linspace(start=self.grid_range[0], end=self.grid_range[1], steps=self.grid_size)] * self.ndim
        return torch.stack(torch.meshgrid(*coords, indexing='ij'), dim=0)

    @property
    def n_layers(self):
        return len(self.modulated_layers)

    def init_grid(self, z):
        # Ensure grid is on the same device as z
        grid_tensor = self.grid.to(z.device)
        return grid_tensor[None].expand(len(z), *([-1] * (self.ndim + 1)))

    def forward(self, z):
        # Original notebook assumes z is [B, latent_dim] and adds spatial dims
        # Let's keep that assumption. If z needs spatial dims, reshape before this.
        if z.dim() == 2:
             # Assume z is [B, latent_dim], add dummy spatial dims for conv
             spatial_dims = (1,) * self.ndim 
             z_conv = z.view(z.shape[0], z.shape[1], *spatial_dims)
        else:
             z_conv = z # Assume z already has spatial dims if not 2D

        # Calculate modulations
        shift_mods = self.shift_modulations(z_conv) if self.shift_modulations is not None else None
        scale_mods = self.scale_modulations(z_conv) if self.scale_modulations is not None else None

        # Split modulations per layer
        shifts_list = list(torch.split(shift_mods, self.split_sizes, dim=1)) if shift_mods is not None else [0.0] * self.n_layers
        scales_list = list(torch.split(scale_mods, self.split_sizes, dim=1)) if scale_mods is not None else [1.0] * self.n_layers

        # Check if the number of splits matches the number of layers
        if len(shifts_list) != self.n_layers or len(scales_list) != self.n_layers:
             raise ValueError(f"Number of modulation splits ({len(shifts_list)}, {len(scales_list)}) " 
                              f"does not match number of modulated layers ({self.n_layers})")

        h = self.init_grid(z)
        for mod_layer, shift, scale in zip(self.modulated_layers, shifts_list, scales_list):
            h = mod_layer(h, shift, scale)
        return self.output_layer(h) # Final output layer (also modulated)
    
class AutoDecoder(nn.Module):
    def __init__(self, latent_dim, num_embeddings, siren_config):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=latent_dim)
        # Pass siren_config dict to SpatialModulationBlock
        self.modulated_siren = SpatialModulationBlock(latent_dim=latent_dim, **siren_config) 

    def forward(self, inputs): # 'inputs' here are the indices for the embedding
        emb = self.embedding(inputs)
        return self.modulated_siren(emb)