from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    # image encoder
    img_size: int = 448
    img_patch_size: int = 32
    img_patch_embedding_size = 10
    embedding_size = 1024

    # imge transformer
    img_transformer_blocks = 3
    img_transformer_hidden_size = 1024
    img_dropout = 0.0

    @property
    def img_w_patches(self) -> int:
        assert self.img_size % self.img_patch_size == 0
        return self.img_size // self.img_patch_size

    @property
    def img_h_patches(self) -> int:
        assert self.img_size % self.img_patch_size == 0
        return self.img_size // self.img_patch_size

    @property
    def img_patches(self) -> int:
        return self.img_w_patches * self.img_h_patches

    @property
    def img_embedding_size(self) -> int:
        return self.img_w_patches * self.img_h_patches * self.img_patch_embedding_size
