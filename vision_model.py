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
            in_channels=3,
            out_channels=config.img_patch_embedding_size,
            kernel_size=config.img_patch_size,
            stride=config.img_patch_size,
            padding="valid",
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
        assert C == 3 or C == 1
        assert H == self.config.img_size
        assert W == self.config.img_size

        # [B x H x W x C] => [B x C x H x W]
        x = x.permute(0, 3, 1, 2)

        # [B x C x H x W] => [B x H_PATCHES x W_PATCHES x IMG_PATCH_EMB]
        x = self.conv(x)

        # [B x H_PATCHES x W_PATCHES x IMG_PATCH_EMB] => [B x PATCHES x IMG_PATCH_EMB]
        x = x.view(B, -1, self.config.img_patch_embedding_size)

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
        B, IMG_PATCHES, IMG_PATH_EMBEDDING = x.size()
        x_clone = x.clone()

        # Normalization
        x = self.norm(x)  # => B x Img_Patches x Img_Patch_Emb

        # Self attention
        qkv = (
            x @ self.qkv
        )  # [B x Img_Patches x Img_Patch_Emb] @ [Img_Patch_Emb x 3-Img_Hidden] => [B x Img_Patches x 3-Img_Hidden]

        # Multihead attention
        qkv = qkv.view(
            B, IMG_PATCHES, self.config.img_multiheads, -1
        )  # B x Img_Patches x HEADS x 3-HEAD_EMBEDDING
        qkv = torch.permute(
            qkv, (0, 2, 1, 3)
        )  # B x HEADS x Img_Patches x 3-HEAD_EMBEDDING
        q, k, v = torch.chunk(
            qkv, 3, dim=3
        )  # => B x HEADS x Img_Patches x HEAD_EMBEDDING

        atten = q @ k.transpose(
            2, 3
        )  # [B x HEADS x Img_Patches x HEAD_EMBEDDING] @ [B x HEADS x HEAD_EMBEDDING x Img_Patches] => [B x HEADS x Img_Patches x Img_Patches]
        x = (
            atten @ v
        )  # [B x HEADS x Img_Patches x Img_Patches] @ [B x HEADS x Img_Patches x HEAD_EMBEDDING] => [B x HEADS x Img_Patches x HEAD_EMBEDDING]
        x = x.transpose(1, 2)  # => [B x Img_Patches x HEADS x HEAD_EMBEDDING]
        x = x.reshape(B, IMG_PATCHES, -1)

        # Dropout
        x = self.dopout(x)

        # FF layer
        x = self.linear(x)
        x = F.relu(x)

        # Residue network
        x = x + x_clone
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
