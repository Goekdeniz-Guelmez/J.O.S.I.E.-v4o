import gzip
import html
import io
import math
from functools import lru_cache
from typing import Callable, List, Optional

from transformers.utils import dummy_torchaudio_objects

import ftfy

import numpy as np
import regex as re
import mlx.core as mx
import mlx.nn as nn

from iopath.common.file_io import g_pathmgr
from timm.models.layers import trunc_normal_

from .helpers import VerboseNNModule


class PatchEmbedGeneric(nn.Module):
    def __init__(self, proj_stem, norm_layer: Optional[nn.Module] = None):
        super().__init__()

        if len(proj_stem) > 1:
            self.proj = nn.Sequential(*proj_stem)
        else:
            self.proj = proj_stem[0]

        self.norm_layer = norm_layer

    def get_patch_layout(self, img_size):
        dummy_img = mx.ones(
            [
                1,
            ]
            + img_size
        )
        dummy_out = self.proj(dummy_img)

        embed_dim = dummy_out.shape[1]
        patches_layout = tuple(dummy_out.shape[2:])
        num_patches = np.prod(patches_layout)

        return patches_layout, num_patches, embed_dim

    def __call__(self, x: mx.array):
        x = self.proj(x).flatten(2).transpose(1, 2) # B C (T_I_V_A.txt) H W -> B (T_I_V_A.txt)HW C

        if self.norm_layer is not None:
            x = self.norm_layer(x)
        return x


class RGBDTPreprocessor(VerboseNNModule):
    def __init__(
        self,
        rgbt_stem: PatchEmbedGeneric,
        depth_stem: PatchEmbedGeneric,
        img_size: List = (3, 224, 224),
        num_cls_tokens: int = 1,
        pos_embed_fn: Callable = None,
        use_type_embed: bool = False,
        init_param_style: str = "openclip",
    ) -> None:
        super().__init__()

        stem = rgbt_stem if rgbt_stem is not None else depth_stem

        # Get the layout, number of patches, and embedding dimension from the stem
        (
            self.patches_layout,
            self.num_patches,
            self.embed_dim,
        ) = stem.get_patch_layout(img_size)

        self.rgbt_stem = rgbt_stem
        self.depth_stem = depth_stem
        self.use_pos_embed = pos_embed_fn is not None
        self.use_type_embed = use_type_embed
        self.num_cls_tokens = num_cls_tokens

        # Initialize positional embedding helper if position embedding function is provided
        if self.use_pos_embed:
            self.pos_embedding_helper = pos_embed_fn(
                patches_layout=self.patches_layout,
                num_cls_tokens=num_cls_tokens,
                num_patches=self.num_patches,
                embed_dim=self.embed_dim,
            )

        # Initialize class tokens if the number of class tokens is greater than 0
        if self.num_cls_tokens > 0:
            self.cls_token = mx.zeros(1, self.num_cls_tokens, self.embed_dim)

        # Initialize type embedding if use_type_embed is True
        if self.use_type_embed:
            self.type_embed = mx.zeros(1, 1, self.embed_dim)

    def tokenize_input_and_cls_pos(self, input, stem, mask):
        # Tokenize input using the stem; tokens will have shape B x L x D
        tokens = stem(input)
        assert tokens.ndim == 3
        assert tokens.shape[2] == self.embed_dim
        B = tokens.shape[0]

        # Add class tokens if num_cls_tokens > 0
        if self.num_cls_tokens > 0:
            class_tokens = self.cls_token.expand(
                B, -1, -1
            )
            tokens = torch.cat((class_tokens, tokens), dim=1)

        # Add positional embeddings if used
        if self.use_pos_embed:
            pos_embed = self.pos_embedding_helper.get_pos_embedding(input, tokens)
            tokens = tokens + pos_embed

        # Add type embeddings if used
        if self.use_type_embed:
            tokens = tokens + self.type_embed.expand(B, -1, -1)

        return tokens

        def __call__(self, vision=None, depth=None, patch_mask=None):
            if patch_mask is not None:
                raise NotImplementedError()

            # Tokenize vision input if provided
            if vision is not None:
                vision_tokens = self.tokenize_input_and_cls_pos(
                    vision, self.rgbt_stem, patch_mask
                )

            # Tokenize depth input if provided
            if depth is not None:
                depth_tokens = self.tokenize_input_and_cls_pos(
                    depth, self.depth_stem, patch_mask
                )

            # Aggregate tokens from vision and depth inputs
            if vision is not None and depth is not None:
                final_tokens = vision_tokens + depth_tokens
            else:
                final_tokens = vision_tokens if vision is not None else depth_tokens

            return_dict = {
                "trunk": {
                    "tokens": final_tokens,
                },
                "head": {},
            }
            return return_dict

class AudioPreprocessor(RGBDTPreprocessor):
    def __init__(self, audio_stem: PatchEmbedGeneric, **kwargs) -> None:
        # Initialize the base class with audio stem
        super().__init__(rgbt_stem=audio_stem, depth_stem=None, **kwargs)

    def __call__(self, audio=None):
        # Forward method for audio input
        return super().__call__(vision=audio)


class ThermalPreprocessor(RGBDTPreprocessor):
    def __init__(self, thermal_stem: PatchEmbedGeneric, **kwargs) -> None:
        # Initialize the base class with thermal stem
        super().__init__(rgbt_stem=thermal_stem, depth_stem=None, **kwargs)

    def __call__(self, thermal=None):
        # Forward method for thermal input
        return super().__call__(vision=thermal)


def build_causal_attention_mask(context_length: int):
    mask = nn.MultiHeadAttention.create_additive_causal_mask(context_length)
    mask = mask.astype(context_length)
    return mask


class TextPreprocessor(VerboseNNModule):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embed_dim: int,
        causal_masking: bool,
        supply_seq_len_to_head: bool = True,
        num_cls_tokens: int = 0,
        init_param_style: str = "openclip",
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = mx.zeros(1, self.context_length + num_cls_tokens, embed_dim)

        self.causal_masking = causal_masking
        if self.causal_masking:
            mask = build_causal_attention_mask(self.context_length)

        self.supply_seq_len_to_head = supply_seq_len_to_head
        self.num_cls_tokens = num_cls_tokens
        self.embed_dim = embed_dim
        if num_cls_tokens > 0:
            assert self.causal_masking is False, "Masking + CLS token isn't implemented"
            self.cls_token = mx.zeros(1, self.num_cls_tokens, embed_dim)


    def __call__(self, text):
        # text tokens are of shape B x L x D
        text_tokens = self.token_embedding(text)
        # concat CLS tokens if any
        if self.num_cls_tokens > 0:
            B = text_tokens.shape[0]
            class_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole class_tokens impl from Phil Wang, thanks
            text_tokens = mx.cat((class_tokens, text_tokens), dim=1)
        text_tokens = text_tokens + self.pos_embed
        return_dict = {
            "trunk": {
                "tokens": text_tokens,
            },
            "head": {},
        }
        # Compute sequence length after adding CLS tokens
        if self.supply_seq_len_to_head:
            text_lengths = text.argmax(dim=-1)
            return_dict["head"] = {
                "seq_len": text_lengths,
            }
        if self.causal_masking:
            return_dict["trunk"].update({"attn_mask": self.mask})
        return return_dict
