import torch
import torch.nn as nn
from einops import rearrange


class PatchEmbedding(nn.Module):
    def __init__(self, image_height, image_width, im_channels,
                 patch_height, patch_width, hidden_size):
        super().__init__()

        assert image_height % patch_height == 0, "Image height must be divisible by patch height"
        assert image_width % patch_width == 0, "Image width must be divisible by patch width"

        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.proj = nn.Conv2d(
            im_channels,
            hidden_size,
            kernel_size=(patch_height, patch_width),
            stride=(patch_height, patch_width),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, 'b d h w -> b (h w) d')
        return x + self.pos_embed
