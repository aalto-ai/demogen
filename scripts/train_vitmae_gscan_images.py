import argparse
import os
import math

# Import pillow and matplotlib first before torch pulls in a different libc
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
from torch.utils.data import DataLoader, Subset, Dataset
from positional_encodings.torch_encodings import (
    PositionalEncoding1D,
    PositionalEncoding2D,
)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
from timm.models.vision_transformer import PatchEmbed, Block

from gscan_metaseq2seq.models.embedding import BOWEmbedding
from gscan_metaseq2seq.util.dataset import PaddingDataset, ReshuffleOnIndexZeroDataset
from gscan_metaseq2seq.util.load_data import load_data, load_data_directories
from gscan_metaseq2seq.util.logging import LoadableCSVLogger
from gscan_metaseq2seq.util.scheduler import transformer_optimizer_config

from gscan_metaseq2seq.util.solver import (
    create_vocabulary,
    create_world,
    state_to_situation,
    reinitialize_world,
)

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class Scheduler(object):
    """Class for keeping track of loss weights and potential noise for the ClevrTex data."""

    def __init__(
        self,
        steps,
        warmup_steps,
        init_mask_ratio,
        init_scale_pixel_ent,
        init_scale_obj_ent,
        init_noise_scale,
        scale_pixel_ent,
        scale_obj_ent,
    ):
        # Define constants
        self.steps = steps
        self.warmup_steps = warmup_steps
        self.delta = steps - warmup_steps
        self.init_mask_ratio = init_mask_ratio
        self.init_scale_pixel_ent = init_scale_pixel_ent
        self.init_scale_obj_ent = init_scale_obj_ent
        self.noise = init_noise_scale

        # Final values of varying paramaters
        self.scale_pixel_ent_final = scale_pixel_ent
        self.scale_obj_ent_final = scale_obj_ent
        self.noise_final = 0.0

        # Initialize values of varying hyperparameters
        self.mask_ratio = init_mask_ratio
        self.scale_pixel_ent = init_scale_pixel_ent
        self.scale_obj_ent = init_scale_obj_ent
        self.noise = init_noise_scale

        self.step_count = 0

    def step(self):
        # If warmup is done, start linear increase of weigths and decrease of mask ration
        self.step_count += 1

        if self.step_count > self.warmup_steps:
            # How many epochs past warmup is it currently
            current = 1.0 * self.step_count - self.warmup_steps

            # Calculate new values of mask ratio and scales for entropy losses
            self.mask_ratio = max(
                self.init_mask_ratio - self.init_mask_ratio * (current / self.delta),
                0.0,
            )
            self.scale_pixel_ent = self.init_scale_pixel_ent + min(
                1.0, current / self.delta
            ) * (self.scale_pixel_ent_final - self.init_scale_pixel_ent)
            self.scale_obj_ent = self.init_scale_obj_ent + min(
                1.0, current / self.delta
            ) * (self.scale_obj_ent_final - self.init_scale_obj_ent)

            # In case noise was used in warmup, it is instantly removed afterwards
            self.noise = self.noise_final

    def values(self):
        # Return the values of the parameters
        return (self.mask_ratio, self.scale_pixel_ent, self.scale_obj_ent, self.noise)


class MaskedAutoencoderViT(pl.LightningModule):
    def __init__(
        self,
        img_resolution,
        n_input_channels,
        patch_size,
        n_latents,
        embed_dim,
        dropout_p,
        nlayers,
        nhead,
        norm_first,
        lr=1e-3,
        wd=1e-2,
        warmup_proportion=0.1,
        decay_power=-1,
        predict_steps=64,
        no_lr_decay=False,
        init_mask_ratio=0.75,
        init_scale_pixel_ent=1e-4,
        init_scale_obj_ent=1e-4,
        scale_pixel_ent=1e-2,
        scale_obj_ent=3e-3,
        init_noise_scale=0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_slots = n_latents
        self.scale = embed_dim**-0.5

        # Embedding and class tokens for object segmentation
        self.patch_embed = PatchEmbed(
            img_resolution, patch_size, n_input_channels, embed_dim
        )
        self.num_patches = self.patch_embed.num_patches
        self.cls_tokens = nn.Parameter(torch.zeros(1, n_latents, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=False
        )  # sin-cos embedding

        # ViT encoder
        self.blocks = nn.ModuleList(
            [
                Block(embed_dim, nhead, 2, qkv_bias=True, norm_layer=nn.LayerNorm)
                for i in range(nlayers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim + 1, embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=False
        )  # sin-cos embedding

        # ViT decoder
        self.decoder_blocks = nn.ModuleList(
            [
                Block(embed_dim, nhead, 2, qkv_bias=True, norm_layer=nn.LayerNorm)
                for i in range(nlayers)
            ]
        )

        # Predicting reconstructions and output masks (alpha channels)
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_pred = nn.Linear(
            embed_dim, patch_size**2 * (n_input_channels + 1), bias=True
        )  # decoder to patch

        # Weight initialization
        self.initialize_weights()

    def initialize_weights(self):
        # sin-cos embedding (not for class tokens)
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=False,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # As in MAE
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initializing cls tokens with small standard deviation
        torch.nn.init.normal_(self.cls_tokens, std=0.002)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def unpatchify_alpha(self, x):
        """
        Copied from original implmentation of unpatchify in MAE,
        but changed channels to 4 to account for the alpha channel.
        x: (N, L, patch_size**2 * 4)
        imgs: (N, 4, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(-1, self.num_slots, h, w, p, p, 4))
        x = torch.einsum("nshwpqc->nschpwq", x)
        imgs = x.reshape(shape=(x.shape[0], self.num_slots, 4, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, noise=0.0):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed

        # Masking as in MAE.
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # Create the k cls tokens
        cls_tokens = self.cls_tokens
        cls_tokens = cls_tokens.expand(x.shape[0], -1, -1)

        # Optional addaition of noise to cls tokens before appending
        if noise > 0.0:
            cls_tokens = cls_tokens + noise * torch.randn_like(cls_tokens)

        # Append the cls_tokens
        x = torch.cat((cls_tokens, x), dim=1)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, ids_restore, mask

    def forward_object_function(self, x):
        # Separate cls tokens from the sequence
        cls_tokens = x[:, : self.num_slots, :]
        x = x[:, self.num_slots :, :]

        # Scaled dot product between embeddings of patches and cls tokens
        attn_logits = torch.bmm(x, cls_tokens.permute(0, 2, 1)) * self.scale

        # Calculate log-softmax as we need logs of softmax later
        attn_logits = attn_logits.log_softmax(dim=-1)
        # Get the softmax for weighted mean
        attn = attn_logits.exp()

        # Compute weights for weighted mean
        w_attn = attn / (attn.sum(dim=1, keepdims=True) + 1e-8)

        # Slots as weighted mean
        slots = torch.bmm(w_attn.permute(0, 2, 1), x)

        return slots, attn_logits, attn

    def forward_broadcast_module(self, slots, attn_logits, ids_restore):
        # Repeat ids for restoring to be one per slot
        # as we decode slots separately
        ids_restore = ids_restore.unsqueeze(1).repeat(1, self.num_slots, 1)
        ids_restore = ids_restore.reshape(-1, ids_restore.size(2))

        # Reshape attention masks for concatenating to broadcasted slots
        attn_logits = attn_logits.permute(0, 2, 1)
        attn_logits = attn_logits.reshape(-1, attn_logits.size(-1), 1)

        # Broadcast the slots
        slots = slots.reshape(-1, 1, slots.size(-1))
        slots = slots.repeat(1, attn_logits.size(-2), 1)

        # Concatenate logs of patch masks to the broadcasted slots
        x = torch.cat((slots, attn_logits), dim=-1)

        # Embed
        x = self.decoder_embed(x)

        # Add mask tokens as in MAE
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1
        )
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(
            x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )

        return x

    def forward_decoder(self, x):
        # Add position embeddings
        x = x + self.decoder_pos_embed

        # Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Predict reconstruction and alpha channel
        x = self.decoder_pred(x)

        # Unpatchify, resulting in reconstructions and output masks (alpha channel)
        x = self.unpatchify_alpha(x)

        # Split into alpha channels (masks) and RGB channels
        out_mask = x[:, :, self.hparams.n_input_channels :, :, :]
        pred_ind = x[:, :, : self.hparams.n_input_channels, :, :]

        # Softmax normalize alpha channels
        out_mask = out_mask.softmax(dim=1)

        # Calucalate full reconstruction as weighted sum
        pred = (out_mask * pred_ind).sum(dim=1)

        return pred, pred_ind, out_mask

    def forward(self, imgs, mask_ratio=0.75, noise=0.0):
        # ViT encoder
        x, ids_restore, mask = self.forward_encoder(imgs, mask_ratio, noise=noise)

        # Object function
        slots, attn_logits, attn = self.forward_object_function(x)

        # Broadcast module
        x = self.forward_broadcast_module(slots, attn_logits, ids_restore)

        # ViT decoder
        pred, pred_ind, out_mask = self.forward_decoder(x)

        return pred, out_mask, pred_ind, slots, attn, mask, ids_restore

    def training_step(self, x, idx):
        query, targets, state = x

        (
            mask_ratio,
            scale_pixel_ent,
            scale_obj_ent,
            noise_scale,
        ) = self.scheduler.values()

        # Forward
        preds, pred_masks, _, _, _, _, _ = self.forward(
            state.permute(0, 3, 1, 2), mask_ratio=mask_ratio, noise=noise_scale
        )

        # Caluculate mean of the output masks for object entropy loss
        mean_masks = pred_masks.mean(dim=(-1, -2))

        # Compute reconstruction loss and entropy losses
        recon_loss = F.mse_loss(preds, state.permute(0, 3, 1, 2))
        pixel_ent = -(pred_masks * torch.log(pred_masks + 1e-8)).sum(dim=1).mean()
        obj_ent = -(mean_masks * torch.log(mean_masks + 1e-8)).sum(dim=1).mean()

        # Full loss as a weighted sum
        loss = recon_loss + scale_pixel_ent * pixel_ent + scale_obj_ent * obj_ent

        self.log("tloss", loss, prog_bar=True)
        self.log("treconst", recon_loss, prog_bar=True)
        self.log("tpixel", pixel_ent, prog_bar=True)
        self.log("tobj", obj_ent, prog_bar=True)

        self.scheduler.step()

        return loss

    def configure_optimizers(self):
        self.scheduler = Scheduler(
            self.trainer.max_steps,
            int(self.trainer.max_steps * self.hparams.warmup_proportion),
            self.hparams.init_mask_ratio,
            self.hparams.init_scale_pixel_ent,
            self.hparams.init_scale_obj_ent,
            self.hparams.init_noise_scale,
            self.hparams.scale_pixel_ent,
            self.hparams.scale_obj_ent,
        )

        return transformer_optimizer_config(
            self,
            self.hparams.lr,
            warmup_proportion=self.hparams.warmup_proportion,
            weight_decay=self.hparams.wd,
            decay_power=self.hparams.decay_power,
            no_lr_decay=self.hparams.no_lr_decay,
        )


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size, orig_size, new_size, new_size)
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                -1, orig_size, orig_size, embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed


class ImageRenderingDemonstrationsDataset(Dataset):
    def __init__(
        self, train_demonstrations, word2idx, colors, nouns, image_downsample=5
    ):
        super().__init__()
        self.train_demonstrations = train_demonstrations
        self.word2idx = word2idx
        self.colors = sorted(colors)
        self.nouns = sorted(nouns)

        vocabulary = create_vocabulary()
        world = create_world(vocabulary)

        self.world = world
        self.vocabulary = vocabulary
        self.image_downsample = image_downsample

    def __getitem__(self, i):
        instruction, actions, state = self.train_demonstrations[i]
        words, situation = state_to_situation(
            instruction, state, self.word2idx, self.colors, self.nouns
        )

        self.world = reinitialize_world(self.world, situation, self.vocabulary)

        img = np.copy(
            self.world.render(mode="rgb_array")[
                :: self.image_downsample, :: self.image_downsample
            ]
            / 255.0
        )

        return instruction, actions, img.astype(np.float32)

    def __len__(self):
        return len(self.train_demonstrations)


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-demonstrations", type=str, required=True)
    parser.add_argument("--valid-demonstrations-directory", type=str, required=True)
    parser.add_argument("--dictionary", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-batch-size", type=int, default=9)
    parser.add_argument("--valid-batch-size", type=int, default=16)
    parser.add_argument("--batch-size-mult", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--nlayers", type=int, default=8)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--norm-first", action="store_true")
    parser.add_argument("--precision", type=int, choices=(16, 32), default=16)
    parser.add_argument("--dropout-p", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no-lr-decay", action="store_true")
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--warmup-proportion", type=float, default=0.1)
    parser.add_argument("--decay-power", type=float, default=-1)
    parser.add_argument("--iterations", type=int, default=2500000)
    parser.add_argument("--disable-shuffle", action="store_true")
    parser.add_argument("--check-val-every", type=int, default=1000)
    parser.add_argument("--limit-val-size", type=int, default=None)
    parser.add_argument("--enable-progress", action="store_true")
    parser.add_argument("--version", type=int, default=None)
    parser.add_argument("--tag", type=str, default="none")
    parser.add_argument("--swa", action="store_true")
    parser.add_argument("--pad-instructions-to", type=int, default=8)
    parser.add_argument("--pad-actions-to", type=int, default=128)
    parser.add_argument("--pad-state-to", type=int, default=36)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--limit-load", type=int, default=None)
    parser.add_argument("--image-downsample", type=int, default=5)
    parser.add_argument("--patch-size", type=int, default=12)
    args = parser.parse_args(argv)

    return args


def main():
    args = parse_args(sys.argv[1:])

    exp_name = "gscan"
    model_name = f"vilbert_img_mae_decode_actions_l_{args.nlayers}_h_{args.nhead}_d_{args.hidden_size}"
    dataset_name = "gscan"
    effective_batch_size = args.train_batch_size * args.batch_size_mult
    exp_name = f"{exp_name}_s_{args.seed}_m_{model_name}_it_{args.iterations}_b_{effective_batch_size}_d_{dataset_name}_t_{args.tag}_drop_{args.dropout_p}"
    model_dir = f"models/{exp_name}/{model_name}"
    model_path = f"{model_dir}/{exp_name}.pt"
    print(model_path)
    print(
        f"Batch size {args.train_batch_size}, mult {args.batch_size_mult}, total {args.train_batch_size * args.batch_size_mult}"
    )

    os.makedirs(model_dir, exist_ok=True)

    seed = args.seed
    iterations = args.iterations

    (
        (
            WORD2IDX,
            ACTION2IDX,
            color_dictionary,
            noun_dictionary,
        ),
        (train_demonstrations, valid_demonstrations_dict),
    ) = load_data_directories(
        args.train_demonstrations, args.dictionary, limit_load=args.limit_load
    )

    IDX2WORD = {i: w for w, i in WORD2IDX.items()}
    IDX2ACTION = {i: w for w, i in ACTION2IDX.items()}

    pad_word = WORD2IDX["[pad]"]
    pad_action = ACTION2IDX["[pad]"]
    sos_action = ACTION2IDX["[sos]"]
    eos_action = ACTION2IDX["[eos]"]

    pl.seed_everything(0)
    train_dataset = ReshuffleOnIndexZeroDataset(
        PaddingDataset(
            ImageRenderingDemonstrationsDataset(
                train_demonstrations, WORD2IDX, color_dictionary, noun_dictionary
            ),
            (args.pad_instructions_to, args.pad_actions_to, None),
            (pad_word, pad_action, None),
        )
    )

    pl.seed_everything(seed)
    # sd = meta_module.state_dict()
    meta_module = MaskedAutoencoderViT(
        72,  # 72x72 images
        3,  # 3 input channels
        args.patch_size,
        16,
        args.hidden_size,
        args.dropout_p,
        args.nlayers,
        args.nhead,
        args.norm_first,
        lr=args.lr,
        decay_power=args.decay_power,
        warmup_proportion=args.warmup_proportion,
        no_lr_decay=args.no_lr_decay,
    )
    # meta_module.load_state_dict(sd)
    print(meta_module)

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)

    check_val_opts = {}
    interval = args.check_val_every / len(train_dataloader)

    # Every check_val_interval steps, regardless of how large the training dataloader is
    if interval > 1.0:
        check_val_opts["check_val_every_n_epoch"] = math.floor(interval)
    else:
        check_val_opts["val_check_interval"] = interval

    checkpoint_cb = ModelCheckpoint(
        monitor="vexact/dataloader_idx_0",
        auto_insert_metric_name=False,
        save_top_k=5,
        mode="max",
    )

    model_name = "masked_ae_vit"
    logs_root_dir = f"{args.log_dir}/{exp_name}/{model_name}/{dataset_name}/{seed}"
    most_recent_version = args.version

    trainer = pl.Trainer(
        logger=[
            TensorBoardLogger(logs_root_dir, version=most_recent_version),
            LoadableCSVLogger(
                logs_root_dir, version=most_recent_version, flush_logs_every_n_steps=10
            ),
        ],
        callbacks=[pl.callbacks.LearningRateMonitor(), checkpoint_cb]
        + (
            [
                StochasticWeightAveraging(
                    swa_lrs=1e-2,
                    annealing_epochs=int(
                        (iterations * args.batch_size_mult)
                        // len(train_dataloader)
                        * 0.2
                    ),
                )
            ]
            if args.swa
            else []
        ),
        max_steps=iterations,
        num_sanity_val_steps=10,
        # gpus=1 if torch.cuda.is_available() else 0,
        # precision=args.precision if torch.cuda.is_available() else 32,
        default_root_dir=logs_root_dir,
        accumulate_grad_batches=args.batch_size_mult,
        gradient_clip_val=0.2,
        **check_val_opts,
    )

    trainer.fit(meta_module, train_dataloader)


if __name__ == "__main__":
    main()
