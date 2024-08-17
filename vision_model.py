import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from typing import OrderedDict


class VisionEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.conv = nn.Conv2d(
            in_channels=3, out_channels=1, kernel_size=3, padding="same"
        )
        self.norm = nn.LayerNorm(
            (self.config.img_patch_size, self.config.img_patch_size)
        )
        self.linear = nn.Linear(
            self.config.img_patch_size**2, self.config.img_patch_embedding_size
        )

        self.pos_embeddings = nn.Embedding(
            num_embeddings=config.img_h_patches * config.img_w_patches,
            embedding_dim=self.config.img_patch_embedding_size,
            max_norm=True,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        x: B x H x W x C
        output: B x Patch-H x Patch-W x Patch-embeeding
        """
        B, H, W, C = x.size()
        assert H == self.config.img_size
        assert W == self.config.img_size

        x = x.view(B, self.config.img_h_patches, self.config.img_patch_size, W, C)
        x = x.view(
            B,
            self.config.img_h_patches,
            self.config.img_patch_size,
            self.config.img_patch_size,
            self.config.img_w_patches,
            C,
        )
        x = x.permute(
            0, 1, 4, 5, 2, 3
        )  # B x H_PATCHES x 16 x 16 x W_PATCHES x C => B x H_PATCHES x W_PATCHES x C x 16 x 16
        x = x.reshape(
            -1, C, self.config.img_patch_size, self.config.img_patch_size
        )  # B x H_PATCHES x W_PATCHES x C x 16 x 16 => B16 x C x 16 x 16
        x = self.norm(x)  # => B16 x C x 16 x 16
        x = self.conv(x)  #  B16 x C x 16 x 16 =>  B16 x 1 x 16 x 16

        x = x.view(-1, self.config.img_patch_size**2)  # => B16 x 256
        x = self.linear(x)  # => B16 x Patch_Emb

        x = x.view(
            B,
            self.config.img_h_patches * self.config.img_w_patches,
            self.config.img_patch_embedding_size,
        )  # => B x H_PATCHES * W_PATCHES x PATCH_EMB

        x = x + self.pos_embeddings(
            torch.arange(
                0,
                self.config.img_h_patches * self.config.img_w_patches,
                dtype=torch.int,
                device=self.device,
            )
        )

        x = x.view(
            B,
            self.config.img_h_patches * self.config.img_w_patches,
            self.config.img_patch_embedding_size,
        )  # => B x Img_Patches x Img_Patch_Emb

        return x


class VisionTransformerBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.norm = nn.LayerNorm((config.img_patches, config.img_patch_embedding_size))
        data = torch.randn(
            config.img_patch_embedding_size, 3 * config.img_transformer_hidden_size
        )  # Img_Patch_Emb x 3-Img_Hidden
        self.qkv = nn.Parameter(data=data)
        self.dopout = nn.Dropout(config.img_dropout)
        self.linear = nn.Linear(
            config.img_transformer_hidden_size, config.img_patch_embedding_size
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        x: B x Img_Patches x Img_Patch_Emb
        output: B x Img_Emb
        """
        # Normalization
        x = self.norm(x)  # => B x Img_Patches x Img_Patch_Emb

        # Self attention
        qkv = (
            x @ self.qkv
        )  # [B x Img_Patches x Img_Patch_Emb] @ [Img_Patch_Emb x 3-Img_Hidden] => [B x Img_Patches x 3-Img_Hidden]
        q, k, v = torch.chunk(qkv, 3, dim=2)  # => B x Img_Patches x Img_Hidden
        atten = q @ k.transpose(
            1, 2
        )  # [B x Img_Patches x Img_Hidden] @ [B x Img_Hidden x Img_Patches] => [B x Img_Patches x Img_Patches]
        x = (
            atten @ v
        )  # [B x Img_Patches x Img_Patches] @ [B x Img_Patches x Img_Hidden] => [B x Img_Patches x Img_Hidden]

        # Dropout
        x = self.dopout(x)

        # FF layer
        x = self.linear(x)
        x = F.relu(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.transformer_blocks = nn.Sequential(
            *[
                VisionTransformerBlock(config=config)
                for i in range(config.img_transformer_blocks)
            ]
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        x: B x Img_Patches x Img_Patch_Emb
        output: B x Img_Emb
        """
        B, IMG_PATCHES, IMG_PATCH_EMBEDDING = x.size()
        x = self.transformer_blocks(x)  # => B x Img_Patches x Img_Patch_Emb
        x = x.view(B, -1)  # => B x Img_Emb
        return x


class VisionModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.vision_encoder = VisionEncoder(config)
        self.vision_transformer = VisionTransformer(config)

    def forward(self, img: torch.tensor) -> torch.tensor:
        """
        img: B x H x W x C
        output: B x Embedding
        """
        x = self.vision_encoder(img)
        x = self.vision_transformer(x)
        return x
