# Adapted from PNDM implmentation (https://github.com/luping-liu/PNDM)
# which is adapted from DDIM implementation (https://github.com/ermongroup/ddim)

import math
import torch
from einops import rearrange
from itertools import pairwise
from torch import nn
from .model import (
    alpha, Attention, ModelMixin, CondSequential, SigmaEmbedderSinCos,
)

# Helper function to add coordinate channels
def add_coords(input_tensor):
    """Adds normalized coordinate channels to the input tensor."""
    B, _, H, W = input_tensor.shape
    yy = torch.arange(H, device=input_tensor.device, dtype=input_tensor.dtype).view(-1, 1).repeat(1, W)
    xx = torch.arange(W, device=input_tensor.device, dtype=input_tensor.dtype).view(1, -1).repeat(H, 1)

    yy = (yy / (H - 1 + 1e-6)) * 2 - 1 # Normalize to [-1, 1], add eps for H=1 or W=1
    xx = (xx / (W - 1 + 1e-6)) * 2 - 1 # Normalize to [-1, 1], add eps for H=1 or W=1

    coords = torch.stack([yy, xx], dim=0).unsqueeze(0).repeat(B, 1, 1, 1) # Shape: [B, 2, H, W]
    return torch.cat([input_tensor, coords], dim=1) # Concatenate along channel dim

class CoordConv2d(nn.Module):
    """A Conv2d layer that optionally adds coordinate channels to the input."""
    def __init__(self, in_channels, out_channels, use_coordconv=False, **kwargs):
        super().__init__()
        self.use_coordconv = use_coordconv
        self.in_channels_actual = in_channels + 2 if use_coordconv else in_channels
        self.conv = nn.Conv2d(self.in_channels_actual, out_channels, **kwargs)

    def forward(self, x):
        if self.use_coordconv:
            x = add_coords(x)
        return self.conv(x)

def Normalize(ch):
    return torch.nn.GroupNorm(num_groups=32, num_channels=ch, eps=1e-6, affine=True)

def Upsample(ch, use_coordconv=False):
    conv = CoordConv2d(ch, ch, kernel_size=3, stride=1, padding=1, use_coordconv=use_coordconv) 
    return nn.Sequential(
        nn.Upsample(scale_factor=2.0, mode='nearest'),
        conv,
    )

def Downsample(ch, use_coordconv=False):
    conv = CoordConv2d(ch, ch, kernel_size=3, stride=2, padding=0, use_coordconv=use_coordconv)
    return nn.Sequential(
        nn.ConstantPad2d((0, 1, 0, 1), 0), # Pad bottom/right for stride=2 conv
        conv,
    )

class ResnetBlock(nn.Module):
    def __init__(self, *, in_ch, out_ch=None, conv_shortcut=False,
                 dropout, temb_channels=512, use_coordconv=False):
        super().__init__()
        self.in_ch = in_ch
        out_ch = in_ch if out_ch is None else out_ch
        self.out_ch = out_ch
        self.use_conv_shortcut = conv_shortcut
        self.use_coordconv = use_coordconv

        self.layer1 = nn.Sequential(
            Normalize(in_ch),
            nn.SiLU(),
            CoordConv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, use_coordconv=self.use_coordconv),
        )
        self.temb_proj = nn.Sequential(
            nn.SiLU(),
            torch.nn.Linear(temb_channels, out_ch),
        )
        self.layer2 = nn.Sequential(
            Normalize(out_ch),
            nn.SiLU(),
            torch.nn.Dropout(dropout),
            CoordConv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, use_coordconv=self.use_coordconv),
        )
        if self.in_ch != self.out_ch:
            if self.use_conv_shortcut:
                kernel_size, stride, padding = 3, 1, 1
            else:
                kernel_size, stride, padding = 1, 1, 0
            self.shortcut = CoordConv2d(in_ch, out_ch,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        use_coordconv=self.use_coordconv)

    def forward(self, x, temb):
        h = x
        h = self.layer1(h)
        h = h + self.temb_proj(temb)[:, :, None, None]
        h = self.layer2(h)
        if self.in_ch != self.out_ch:
            x = self.shortcut(x)
        return x + h

class AttnBlock(nn.Module):
    def __init__(self, ch, num_heads=1):
        super().__init__()
        # Normalize input along the channel dimension
        self.norm = Normalize(ch)
        # Attention over D: (B, N, D) -> (B, N, D)
        self.attn = Attention(head_dim=ch // num_heads, num_heads=num_heads)
        # Apply 1x1 convolution for projection
        self.proj_out = nn.Conv2d(ch, ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        # temb is currently not used, but included for CondSequential to work
        B, C, H, W = x.shape
        h_ = self.norm(x)
        h_ = rearrange(h_, 'b c h w -> b (h w) c')
        h_ = self.attn(h_)
        h_ = rearrange(h_, 'b (h w) c -> b c h w', h=H, w=W)
        return x + self.proj_out(h_)

class Unet(nn.Module, ModelMixin):
    def __init__(self, in_dim, in_ch, out_ch,
                 ch               = 128,
                 ch_mult          = (1,2,2,2),
                 embed_ch_mult    = 4,
                 num_res_blocks   = 2,
                 attn_resolutions = (16,),
                 dropout          = 0.1,
                 resamp_with_conv = True,
                 sig_embed        = None,
                 cond_embed       = None,
                 use_coordconv    = False,
                 ):
        super().__init__()

        self.ch = ch
        self.in_dim = in_dim
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.input_dims = (in_ch, in_dim, in_dim)
        self.temb_ch = self.ch * embed_ch_mult
        self.use_coordconv = use_coordconv

        # Embeddings
        self.sig_embed = sig_embed or SigmaEmbedderSinCos(self.temb_ch)
        make_block = lambda in_ch, out_ch: ResnetBlock(
            in_ch=in_ch, out_ch=out_ch, temb_channels=self.temb_ch, dropout=dropout, use_coordconv=self.use_coordconv
        )
        self.cond_embed = cond_embed

        # Downsampling
        curr_res = in_dim
        in_ch_dim = [ch * m for m in (1,)+ch_mult]
        self.conv_in = CoordConv2d(in_ch, self.ch, kernel_size=3, stride=1, padding=1, use_coordconv=self.use_coordconv)
        self.downs = nn.ModuleList()
        for i, (block_in, block_out) in enumerate(pairwise(in_ch_dim)):
            down = nn.Module()
            down.blocks = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                block = [make_block(block_in,block_out)]
                if curr_res in attn_resolutions:
                    block.append(AttnBlock(block_out))
                down.blocks.append(CondSequential(*block))
                block_in = block_out
            if i < self.num_resolutions - 1: # Not last iter
                down.downsample = Downsample(block_in, use_coordconv=self.use_coordconv)
                curr_res = curr_res // 2
            self.downs.append(down)

        # Middle
        self.mid = CondSequential(
            make_block(block_in, block_in),
            AttnBlock(block_in),
            make_block(block_in, block_in)
        )

        # Upsampling
        self.ups = nn.ModuleList()
        for i_level, (block_out, next_skip_in) in enumerate(pairwise(reversed(in_ch_dim))):
            up = nn.Module()
            up.blocks = nn.ModuleList()
            skip_in = block_out
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = next_skip_in
                block = [make_block(block_in+skip_in, block_out)]
                if curr_res in attn_resolutions:
                    block.append(AttnBlock(block_out))
                up.blocks.append(CondSequential(*block))
                block_in = block_out
            if i_level < self.num_resolutions - 1: # Not last iter
                up.upsample = Upsample(block_in, use_coordconv=self.use_coordconv)
                curr_res = curr_res * 2
            self.ups.append(up)

        # Out
        self.out_layer = nn.Sequential(
            Normalize(block_in),
            nn.SiLU(),
            nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, sigma, cond=None):
        assert x.shape[2] == x.shape[3] == self.in_dim

        # Embeddings
        emb = self.sig_embed(x.shape[0], sigma.squeeze())
        if self.cond_embed is not None:
            assert cond is not None and x.shape[0] == cond.shape[0], \
                'Conditioning must have same batches as x!'
            emb += self.cond_embed(cond)

        # downsampling
        hs = [self.conv_in(x)]
        for down in self.downs:
            for block in down.blocks:
                h = block(hs[-1], emb)
                hs.append(h)
            if hasattr(down, 'downsample'):
                hs.append(down.downsample(hs[-1]))

        # middle
        h = self.mid(hs[-1], emb)

        # upsampling
        for up in self.ups:
            for block in up.blocks:
                h = block(torch.cat([h, hs.pop()], dim=1), emb)
            if hasattr(up, 'upsample'):
                h = up.upsample(h)

        # out
        return self.out_layer(h)
