import argparse
import os
import numpy as np
import math

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
import sys
from torch.utils.data import DataLoader, Subset
from positional_encodings.torch_encodings import PositionalEncoding1D
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from positional_encodings.torch_encodings import (
    PositionalEncoding1D,
    PositionalEncoding2D,
)

from gscan_metaseq2seq.models.embedding import BOWEmbedding
from gscan_metaseq2seq.util.dataset import (
    PaddingDataset,
    ReshuffleOnIndexZeroDataset,
    MapDataset,
    ReorderSupportsByDistanceDataset,
)
from gscan_metaseq2seq.util.load_data import load_data, load_data_directories
from gscan_metaseq2seq.util.logging import LoadableCSVLogger
from gscan_metaseq2seq.util.scheduler import transformer_optimizer_config
from gscan_metaseq2seq.util.padding import pad_to


def init_parameters(module, scale=1e-2):
    if type(module) in [nn.LayerNorm]:
        return

    if type(module) in [nn.MultiheadAttention]:
        torch.nn.init.normal_(module.in_proj_weight, 0, scale)
        return

    if type(module) in [nn.Conv2d]:
        return

    if getattr(module, "weight", None) is not None:
        torch.nn.init.normal_(module.weight, 0, scale)

    if getattr(module, "bias", None) is not None:
        torch.nn.init.zeros_(module.bias)


class ImagePatchEncoding(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        patches = self.conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        return patches.flatten(1, 2)


class DictEmbedding(nn.Module):
    def __init__(self, keys, embed_dim, padding_key=None):
        super().__init__()
        for i, key in enumerate(keys):
            self.register_buffer(key, torch.tensor(i, dtype=torch.long))
        self.embedding = nn.Embedding(
            len(keys), embed_dim, padding_idx=getattr(self, padding_key).item()
        )

    def __getitem__(self, key):
        return self.embedding(getattr(self, key))


def autoregressive_model_unroll_predictions(
    model, inputs, target, sos_target_idx, eos_target_idx, pad_target_idx
):
    with torch.inference_mode(), torch.autocast(device_type=str(target.device).split(":")[0], dtype=torch.float16, enabled=True):
        encodings, key_padding_mask = model.encode(*inputs)

    # Recursive decoding, start with a batch of SOS tokens
    decoder_in = torch.tensor(sos_target_idx, dtype=torch.long, device=model.device)[
        None
    ].expand(target.shape[0], 1)

    logits = []

    with torch.inference_mode(), torch.autocast(device_type=str(target.device).split(":")[0], dtype=torch.float16, enabled=True):
        for i in trange(target.shape[1], desc="Gen tgts"):
            stopped_mask = (decoder_in == eos_target_idx).any(dim=-1)
            still_going_mask = ~stopped_mask
            still_going_indices = torch.nonzero(still_going_mask).flatten()

            if still_going_mask.any(dim=-1):
                decoder_in_still_going = decoder_in[still_going_mask]
                encodings_still_going = encodings.transpose(0, 1)[still_going_mask].transpose(0, 1)
                key_padding_mask_still_going = key_padding_mask[still_going_mask]

                current_logits = model.decode_autoregressive(
                    decoder_in_still_going,
                    encodings_still_going,
                    key_padding_mask_still_going
                )[:, -1]

                scatter_target = torch.zeros_like(current_logits[0, None, :].expand(encodings.shape[1], current_logits.shape[1]))
                scatter_target.scatter_(
                    0,
                    still_going_indices[:, None].expand(still_going_indices.shape[0], current_logits.shape[1]),
                    current_logits
                )
                logits.append(scatter_target)
            else:
                logits.append(logits[-1].clone())

            decoder_out = logits[-1].argmax(dim=-1)
            decoder_in = torch.cat([decoder_in, decoder_out[:, None]], dim=1)

        decoded = decoder_in
        logits = torch.stack(logits, dim=1)

        # these are shifted off by one
        decoded_eq_mask = (
            (decoded == eos_target_idx).int().cumsum(dim=-1).bool()[:, :-1]
        )
        decoded_eq_mask = torch.cat([
            torch.zeros_like(decoded_eq_mask[:, :1]),
            decoded_eq_mask
        ], dim=-1)
        decoded[decoded_eq_mask] = pad_target_idx
        decoded = decoded[:, 1:]

    exacts = (decoded == target).all(dim=-1).cpu().numpy()

    return ([
        decoded,
        logits,
        exacts,
        target
    ])


class BigSymbolTransformerLearner(pl.LightningModule):
    def __init__(
        self,
        n_state_components,
        in_vocab_size,
        out_vocab_size,
        embed_dim,
        dropout_p,
        nlayers,
        nhead,
        pad_word_idx,
        pad_action_idx,
        sos_action_idx,
        eos_action_idx,
        pad_state_idx,
        norm_first=False,
        lr=1e-4,
        wd=1e-2,
        warmup_proportion=0.001,
        decay_power=-1,
        predict_steps=64,
        metalearn_dropout_p=0.0,
        metalearn_include_permutations=False,
        max_context_size=1024,
        need_support_states=False
    ):
        super().__init__()
        self.bow_embedding = nn.Sequential(
            BOWEmbedding(
                64, n_state_components, embed_dim
            ),
            nn.Linear(n_state_components * embed_dim, embed_dim)
        )
        self.in_embedding = nn.Embedding(
            in_vocab_size, embed_dim, padding_idx=pad_word_idx
        )
        self.out_embedding = nn.Embedding(
            out_vocab_size, embed_dim, padding_idx=pad_action_idx
        )
        self.pos_encoding = PositionalEncoding1D(embed_dim)
        self.special_tokens = DictEmbedding(
            [
                "pad",
                "sos",
                "eos",
                "sep_state_support",
                "sep_instr_support",
                "sep_target_support",
                "sep_query_state",
                "sep_query_instr",
                "sep_query_end",
            ],
            embed_dim,
            padding_key="pad",
        )
        self.dropout = nn.Dropout(p=dropout_p)
        self.norm = nn.LayerNorm(embed_dim)
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=nhead,
            dropout=dropout_p,
            norm_first=norm_first,
            activation=F.silu,
            num_encoder_layers=nlayers,
            num_decoder_layers=nlayers,
        )
        self.out = nn.Linear(embed_dim, out_vocab_size)
        self.pad_word_idx = pad_word_idx
        self.pad_action_idx = pad_action_idx
        self.sos_action_idx = sos_action_idx
        self.eos_action_idx = eos_action_idx
        self.pad_state_idx = pad_state_idx

        self.apply(init_parameters)
        self.save_hyperparameters()

    def configure_optimizers(self):
        return transformer_optimizer_config(
            self,
            self.hparams.lr,
            warmup_proportion=self.hparams.warmup_proportion,
            weight_decay=self.hparams.wd,
            decay_power=self.hparams.decay_power,
            optimizer_kwargs={"fused": True},
        )

    def encode(self, support_state, x_supports, y_supports, queries):
        return self.encoder(support_state, x_supports, y_supports, queries)

    def decode_autoregressive(self, decoder_in, encoder_outputs, encoder_padding):
        return self.decoder(decoder_in, encoder_outputs, encoder_padding)

    def assemble_multimodal_inputs(
        self,
        query_state,
        support_states,
        query_instruction,
        query_decoder_in,
        support_instructions,
        support_targets,
    ):
        # We have to do some fairly heavy lifting in the forward function
        # to assemble the input sequence from the multimodal inputs
        encoded_query_state_img = self.bow_embedding(query_state.float())
        support_img_padding = (support_states == self.pad_state_idx).all(dim=-1)
        query_state_padding = (query_state == self.pad_state_idx).all(dim=-1)
        encoded_support_state_img = self.bow_embedding(
            support_states.float().flatten(0, 1)
        ).unflatten(0, (support_states.shape[0], support_states.shape[1]))

        encoded_query_instruction = self.in_embedding(query_instruction)
        encoded_support_instructions = self.in_embedding(support_instructions)
        encoded_support_outputs = self.out_embedding(support_targets)
        encoded_query_decoder_in = self.out_embedding(query_decoder_in)

        # Now that we have everything, we have to select elements according
        # to what is not padded, and try to keep things in batches
        #
        # The basic idea is that we assemble everything into a big sequence
        # first and also keep track of a big "padding sequence" and its inverse.

        # B x S x L x E => B x (S x L) x E
        context_in = torch.cat(
            [
                torch.cat(
                    [
                        self.special_tokens["sep_state_support"][None, None, None]
                        .expand(
                            encoded_support_state_img.shape[0],
                            encoded_support_state_img.shape[1],
                            1,
                            -1,
                        )
                        .clone(),
                        *([encoded_support_state_img] if self.hparams.need_support_states else []),
                        self.special_tokens["sep_instr_support"][None, None, None]
                        .expand(
                            encoded_support_state_img.shape[0],
                            encoded_support_state_img.shape[1],
                            1,
                            -1,
                        )
                        .clone(),
                        encoded_support_instructions,
                        self.special_tokens["sep_target_support"][None, None, None]
                        .expand(
                            encoded_support_state_img.shape[0],
                            encoded_support_state_img.shape[1],
                            1,
                            -1,
                        )
                        .clone(),
                        encoded_support_outputs,
                    ],
                    dim=-2,
                ).flatten(1, 2),
                self.special_tokens["sep_query_state"][None, None].expand(
                    encoded_query_state_img.shape[0], 1, -1
                ),
                encoded_query_state_img,
                self.special_tokens["sep_query_instr"][None, None].expand(
                    encoded_query_instruction.shape[0], 1, -1
                ),
                encoded_query_instruction,
                self.special_tokens["sep_query_end"][None, None].expand(
                    encoded_query_instruction.shape[0], 1, -1
                ),
            ],
            dim=-2,
        )

        sep_nopad_token = torch.zeros_like(
            self.special_tokens["sep_state_support"][..., 0][
                None, None, None
            ]
            .expand(
                encoded_support_state_img.shape[0],
                encoded_support_state_img.shape[1],
                1,
            )
            .clone()
        ).bool()

        # Now we make the corresponding padding sequence. A value
        # of True means padded
        #
        # B x S x L => B x (S x L)
        padding_sequence = torch.cat(
            [
                torch.cat(
                    [
                        sep_nopad_token,
                        *([support_img_padding] if self.hparams.need_support_states else []),
                        sep_nopad_token,
                        support_instructions == self.pad_word_idx,
                        sep_nopad_token,
                        support_targets == self.pad_action_idx,
                    ],
                    dim=-1,
                ).flatten(1, 2),
                sep_nopad_token[:, 0],
                query_state_padding,
                sep_nopad_token[:, 0],
                query_instruction == self.pad_word_idx,
                sep_nopad_token[:, 0],
            ],
            dim=-1,
        )
        inv_padding_sequence = ~padding_sequence

        # Now that we have the padding sequences, we can assemble
        # the selection indices by taking the cumulative sum of
        # padding - 1
        non_padding_target_idx = (
            inv_padding_sequence.int().cumsum(dim=-1) - 1
        ) * inv_padding_sequence.int()

        # All the padding goes to indices that are now offset by the max
        # value of the non-padding selection index + 1, eg, if the
        # non-padding ended at index 10, then padding starts at index 11
        padding_target_idx = (
            padding_sequence.int().cumsum(dim=-1)
            - 1
            + (non_padding_target_idx.max(dim=-1).values[..., None] + 1)
        ) * padding_sequence.int()

        # Add them together to get the target index. This is
        # where we want all the elements to go in the padding_idx
        # so that the padding elements go to the end.
        target_idx = non_padding_target_idx + padding_target_idx

        # Now we have to argsort the target_idx to get the reselection_idx.
        # This tells us the order in which to select elements from the
        # original sequence such that we get all the non-padding elements
        # first and then all the padding at the end. We can use torch.gather
        # with this.
        reselection_idx = target_idx.argsort(dim=-1)

        # Assign a single padding value to all the padding in the
        # original tensor
        context_in = (
            context_in * inv_padding_sequence[..., None].float()
            + self.special_tokens["pad"][None, None].expand(
                context_in.shape[0], context_in.shape[1], -1
            )
            * padding_sequence[..., None].float()
        )

        # Now we can select from the original tensor according to the
        # the reselection_idx
        context_in = torch.gather(
            context_in,
            -2,
            reselection_idx[..., None].expand(
                reselection_idx.shape[0], reselection_idx.shape[1], context_in.shape[-1]
            ),
        )
        context_pad = torch.gather(padding_sequence, -1, reselection_idx)

        # Now we can truncate the context_in according to the
        # max_context_size
        context_in = context_in[..., : self.hparams.max_context_size, :]
        context_pad = context_pad[..., : self.hparams.max_context_size]

        return (
            context_in,
            context_pad,
            encoded_query_decoder_in,
            query_decoder_in == self.pad_action_idx,
        )

    def forward(self, x):
        (
            query_state_img,
            query_instruction,
            query_decoder_in,
            support_state_imgs,
            support_instructions,
            support_targets,
        ) = x
        (
            context_in,
            context_pad,
            decoder_in,
            decoder_pad,
        ) = self.assemble_multimodal_inputs(
            query_state_img,
            query_instruction,
            query_decoder_in,
            support_state_imgs,
            support_instructions,
            support_targets,
        )

        # Now we can add positional encodings to everything
        # and do the usual norm + dropout
        context_in = context_in + self.pos_encoding(context_in)
        context_in = self.dropout(self.norm(context_in))
        decoder_in = decoder_in + self.pos_encoding(decoder_in)
        decoder_in = self.dropout(self.norm(decoder_in))

        # Regular old transformer
        decoded_sequence = self.transformer(
            src=context_in.transpose(0, 1),
            tgt=decoder_in.transpose(0, 1),
            src_key_padding_mask=context_pad,
            memory_key_padding_mask=context_pad,
            tgt_key_padding_mask=decoder_pad,
            tgt_mask=nn.Transformer.generate_square_subsequent_mask(
                decoder_pad.shape[-1]
            )
            .bool()
            .to(decoder_in.device),
        ).transpose(0, 1)

        return self.out(decoded_sequence)

    def training_step(self, x, idx):
        (
            query_state_img,
            support_state_imgs,
            query_instruction,
            targets,
            support_instructions,
            support_targets,
        ) = x

        decoder_in = torch.cat(
            [torch.ones_like(targets)[:, :1] * self.sos_action_idx, targets],
            dim=-1,
        )[:, :-1]

        preds = self.forward(
            (
                query_state_img,
                support_state_imgs,
                query_instruction,
                decoder_in,
                support_instructions,
                support_targets,
            )
        )

        actions_mask = targets == self.pad_action_idx

        # Ultimately we care about the cross entropy loss
        loss = F.cross_entropy(
            preds.flatten(0, -2),
            targets.flatten().long(),
            ignore_index=self.pad_action_idx,
        )

        argmax_preds = preds.argmax(dim=-1)
        argmax_preds[actions_mask] = self.pad_action_idx
        exacts = (argmax_preds == targets).all(dim=-1).to(torch.float).mean()

        self.log("tloss", loss, prog_bar=True)
        self.log("texact", exacts, prog_bar=True)
        self.log(
            "tacc",
            (preds.argmax(dim=-1)[~actions_mask] == targets[~actions_mask])
            .float()
            .mean(),
            prog_bar=True,
        )

        return loss

    def validation_step(self, x, idx, dataloader_idx=0):
        (
            query_state_img,
            support_state_imgs,
            query_instruction,
            targets,
            support_instructions,
            support_targets,
        ) = x

        decoder_in = torch.cat(
            [torch.ones_like(targets)[:, :1] * self.sos_action_idx, targets],
            dim=-1,
        )[:, :-1]

        preds = self.forward(
            (
                query_state_img,
                support_state_imgs,
                query_instruction,
                decoder_in,
                support_instructions,
                support_targets,
            )
        )

        actions_mask = targets == self.pad_action_idx

        # Ultimately we care about the cross entropy loss
        loss = F.cross_entropy(
            preds.flatten(0, -2),
            targets.flatten().long(),
            ignore_index=self.pad_action_idx,
        )

        argmax_preds = preds.argmax(dim=-1)
        argmax_preds[actions_mask] = self.pad_action_idx
        exacts = (argmax_preds == targets).all(dim=-1).to(torch.float).mean()

        self.log("vloss", loss, prog_bar=True)
        self.log("vexact", exacts, prog_bar=True)
        self.log(
            "vacc",
            (preds.argmax(dim=-1)[~actions_mask] == targets[~actions_mask])
            .float()
            .mean(),
            prog_bar=True,
        )

        return loss

    def predict_step(self, x, idx, dl_idx=0):
        (
            query_state_img,
            support_state_imgs,
            query_instruction,
            targets,
            support_instructions,
            support_targets,
        ) = x
        (
            context_in,
            context_pad,
            _,
            _,
        ) = self.assemble_multimodal_inputs(
            query_state_img,
            support_state_imgs,
            query_instruction,
            targets,
            support_instructions,
            support_targets,
        )

        # Now we can add positional encodings to everything
        # and do the usual norm + dropout
        context_in = context_in + self.pos_encoding(context_in)
        context_in = self.dropout(self.norm(context_in))

        encoded_sequence = self.transformer.encoder(
            src=context_in.transpose(0, 1),
            src_key_padding_mask=context_pad
        )

        decoder_in = torch.tensor([self.sos_action_idx])[None].expand(
            context_in.shape[0], 1
        ).cuda()
        logits = []

        for i in range(targets.shape[1]):
            stopped_mask = (decoder_in == self.eos_action_idx).any(dim=-1)
            still_going_mask = ~stopped_mask
            still_going_indices = torch.nonzero(still_going_mask).flatten()

            if still_going_mask.any(dim=-1):
                decoder_in_still_going = decoder_in[still_going_mask]
                encodings_still_going = encoded_sequence.transpose(0, 1)[still_going_mask].transpose(0, 1)
                key_padding_mask_still_going = context_pad[still_going_mask]

                decoder_embeddings = self.out_embedding(decoder_in_still_going)
                decoder_embeddings = decoder_embeddings + self.pos_encoding(decoder_embeddings)
                decoder_embeddings = self.dropout(self.norm(decoder_embeddings))

                current_logits = self.out(self.transformer.decoder(
                    tgt=decoder_embeddings.transpose(0, 1),
                    memory=encodings_still_going,
                    tgt_key_padding_mask=(decoder_in_still_going == self.pad_action_idx),
                    memory_key_padding_mask=key_padding_mask_still_going
                )[-1])

                scatter_target = torch.zeros_like(current_logits[0, None, :].expand(encoded_sequence.shape[1], current_logits.shape[1]))
                scatter_target.scatter_(
                    0,
                    still_going_indices[:, None].expand(still_going_indices.shape[0], current_logits.shape[1]),
                    current_logits
                )
                logits.append(scatter_target)
            else:
                logits.append(logits[-1].clone())

            decoder_out = logits[-1].argmax(dim=-1)
            decoder_in = torch.cat([decoder_in, decoder_out[:, None]], dim=1)

        decoded = decoder_in
        logits = torch.stack(logits, dim=1)

        decoded_eq_mask = (
            (decoder_in == self.eos_action_idx).int().cumsum(dim=-1).bool()[:, :-1]
        )
        decoded = decoder_in[:, 1:]
        decoded[decoded_eq_mask] = self.pad_action_idx

        exacts = (decoded == targets).all(dim=-1)

        return decoded, logits, exacts, targets


class PermuteActionsDataset(Dataset):
    def __init__(
        self,
        dataset,
        x_categories,
        y_categories,
        pad_word_idx,
        pad_action_idx,
        shuffle=True,
        seed=0,
    ):
        super().__init__()
        self.dataset = dataset
        self.x_categories = x_categories
        self.y_categories = y_categories
        self.pad_word_idx = pad_word_idx
        self.pad_action_idx = pad_action_idx
        self.shuffle = shuffle
        self.generator = np.random.default_rng(seed)

    def state_dict(self):
        return {"random_state": self.generator.__getstate__()}

    def load_state_dict(self, sd):
        if "random_state" in sd:
            self.generator.__setstate__(sd["random_state"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        (
            query_state,
            support_state,
            queries,
            targets,
            x_supports,
            y_supports,
        ) = self.dataset[idx]

        x_permutation = np.arange(self.x_categories)
        y_permutation = np.arange(self.y_categories)

        # Compute permutations of outputs
        if self.shuffle:
            # Do the permutation
            x_permutation[0 : self.pad_word_idx] = x_permutation[0 : self.pad_word_idx][
                self.generator.permutation(self.pad_word_idx)
            ]
            y_permutation[0 : self.pad_action_idx] = y_permutation[
                0 : self.pad_action_idx
            ][self.generator.permutation(self.pad_action_idx)]

            # Only permute the outputs, not the inputs
            y_supports = [y_permutation[np.array(ys)] for ys in y_supports]
            targets = y_permutation[np.array(targets)]

        return (
            query_state,
            support_state,
            queries,
            targets,
            x_supports,
            y_supports,
        )


class ShuffleDemonstrationsDataset(Dataset):
    def __init__(self, dataset, active, seed=0):
        super().__init__()
        self.dataset = dataset
        self.active = active
        self.generator = np.random.default_rng(seed)

    def state_dict(self):
        return {"random_state": self.generator.__getstate__()}

    def load_state_dict(self, sd):
        if "random_state" in sd:
            self.generator.__setstate__(sd["random_state"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if not self.active:
            return self.dataset[idx]

        (
            query_state,
            support_state,
            queries,
            targets,
            x_supports,
            y_supports,
        ) = self.dataset[idx]

        support_permutation = self.generator.permutation(len(x_supports))

        return (
            query_state,
            [support_state[i] for i in support_permutation]
            if isinstance(support_state, list)
            else support_state,
            queries,
            targets,
            [x_supports[i] for i in support_permutation]
            if isinstance(x_supports, list)
            else x_supports[support_permutation],
            [y_supports[i] for i in support_permutation]
            if isinstance(y_supports, list)
            else y_supports[support_permutation],
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-demonstrations", type=str, required=True)
    parser.add_argument("--valid-demonstrations-directory", type=str, required=True)
    parser.add_argument("--dictionary", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-batch-size", type=int, default=1024)
    parser.add_argument("--valid-batch-size", type=int, default=128)
    parser.add_argument("--batch-size-mult", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--nlayers", type=int, default=8)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--norm-first", action="store_true")
    parser.add_argument("--dropout-p", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--warmup-proportion", type=float, default=0.1)
    parser.add_argument("--decay-power", type=int, default=-1)
    parser.add_argument("--iterations", type=int, default=2500000)
    parser.add_argument("--disable-shuffle", action="store_true")
    parser.add_argument("--check-val-every", type=int, default=500)
    parser.add_argument("--limit-val-size", type=int, default=None)
    parser.add_argument("--enable-progress", action="store_true")
    parser.add_argument("--restore-from-checkpoint", action="store_true")
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default="gscan")
    parser.add_argument("--tag", type=str, default="none")
    parser.add_argument("--metalearn-dropout-p", type=float, default=0.0)
    parser.add_argument("--metalearn-demonstrations-limit", type=int, default=6)
    parser.add_argument("--metalearn-include-permutations", action="store_true")
    parser.add_argument("--pad-instructions-to", type=int, default=8)
    parser.add_argument("--pad-actions-to", type=int, default=128)
    parser.add_argument("--pad-state-to", type=int, default=36)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--limit-load", type=int, default=None)
    parser.add_argument("--dataloader-ncpus", type=int, default=1)
    parser.add_argument("--shuffle-demonstrations", action="store_true")
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--image-downsample", type=int, default=5)
    parser.add_argument("--patch-size", type=int, default=12)
    parser.add_argument("--max-context-size", type=int, default=1024)
    parser.add_argument("--need-support-states", action="store_true")
    args = parser.parse_args()

    exp_name = "meta_gscan"
    model_name = f"meta_symbol_encdec_big_transformer_l_{args.nlayers}_h_{args.nhead}_d_{args.hidden_size}"
    dataset_name = args.dataset_name
    effective_batch_size = args.train_batch_size * args.batch_size_mult
    exp_name = f"{exp_name}_s_{args.seed}_m_{model_name}_it_{args.iterations}_b_{effective_batch_size}_d_{dataset_name}_t_{args.tag}_drop_{args.dropout_p}_ml_d_limit_{args.metalearn_demonstrations_limit}"
    model_dir = f"models/{exp_name}/{model_name}"
    model_path = f"{model_dir}/{exp_name}.pt"
    print(model_path)
    print(
        f"Batch size {args.train_batch_size}, mult {args.batch_size_mult}, total {args.train_batch_size * args.batch_size_mult}"
    )

    os.makedirs(model_dir, exist_ok=True)

    if os.path.exists(f"{model_path}"):
        print(f"Skipping {model_path} as it already exists")
        return

    torch.set_float32_matmul_precision("medium")
    print("Flash attention:", torch.backends.cuda.flash_sdp_enabled())

    seed = args.seed
    iterations = args.iterations

    (
        (
            WORD2IDX,
            ACTION2IDX,
            color_dictionary,
            noun_dictionary,
        ),
        (meta_train_demonstrations, meta_valid_demonstrations_dict),
    ) = load_data_directories(
        args.train_demonstrations, args.dictionary, limit_load=args.limit_load
    )

    IDX2WORD = {i: w for w, i in WORD2IDX.items()}
    IDX2ACTION = {i: w for w, i in ACTION2IDX.items()}

    pad_word = WORD2IDX["[pad]"]
    pad_action = ACTION2IDX["[pad]"]
    sos_action = ACTION2IDX["[sos]"]
    eos_action = ACTION2IDX["[eos]"]
    pad_state = 0

    pl.seed_everything(0)
    meta_train_dataset = ReshuffleOnIndexZeroDataset(
        PaddingDataset(
            PermuteActionsDataset(
                ShuffleDemonstrationsDataset(
                    ReorderSupportsByDistanceDataset(
                        MapDataset(
                            MapDataset(
                                meta_train_demonstrations,
                                lambda x: (
                                    x[2],
                                    x[3],
                                    x[0],
                                    x[1],
                                    x[4],
                                    x[5],
                                    x[6],
                                ),
                            ),
                            lambda x: (
                                x[0],
                                [x[1]] * len(x[-1])
                                if not isinstance(x[1][0], list)
                                else x[1],
                                x[2],
                                x[3],
                                x[4],
                                x[5],
                                x[6],
                            ),
                        ),
                        args.metalearn_demonstrations_limit,
                    ),
                    args.shuffle_demonstrations,
                ),
                len(WORD2IDX),
                len(ACTION2IDX),
                pad_word,
                pad_action,
                shuffle=not args.disable_shuffle,
            ),
            (
                (args.pad_state_to, None),
                (args.metalearn_demonstrations_limit, args.pad_state_to, None),
                32,
                128,
                (args.metalearn_demonstrations_limit, 32),
                (args.metalearn_demonstrations_limit, 128),
            ),
            (pad_state, pad_state, pad_word, pad_action, pad_word, pad_action),
        )
    )

    pl.seed_everything(seed)
    meta_module = BigSymbolTransformerLearner(
        7,
        len(WORD2IDX),
        len(ACTION2IDX),
        args.hidden_size,
        args.dropout_p,
        args.nlayers,
        args.nhead,
        pad_word,
        pad_action,
        sos_action,
        eos_action,
        pad_state,
        norm_first=args.norm_first,
        lr=args.lr,
        decay_power=args.decay_power,
        warmup_proportion=args.warmup_proportion,
        metalearn_dropout_p=args.metalearn_dropout_p,
        metalearn_include_permutations=args.metalearn_include_permutations,
        max_context_size=args.max_context_size,
        need_support_states=args.need_support_states
    )
    print(meta_module)

    pl.seed_everything(0)
    train_dataloader = DataLoader(
        meta_train_dataset,
        batch_size=args.train_batch_size,
        num_workers=1,  # args.dataloader_ncpus,
        prefetch_factor=4,
    )

    check_val_opts = {}
    interval = args.check_val_every / len(train_dataloader)

    # Every check_val_interval steps, regardless of how large the training dataloader is
    if interval > 1.0:
        check_val_opts["check_val_every_n_epoch"] = math.floor(interval)
    else:
        check_val_opts["val_check_interval"] = interval

    logs_root_dir = f"{args.log_dir}/{exp_name}/{model_name}/{dataset_name}/{seed}"
    most_recent_version = args.version

    strategy_kwargs = (
        {
            "strategy": pl.strategies.FSDPStrategy(
                activation_checkpointing=(
                    [nn.TransformerEncoderLayer, nn.TransformerDecoderLayer]
                ),
            )
        }
        if args.activation_checkpointing
        else {}
    )

    meta_trainer = pl.Trainer(
        logger=[
            TensorBoardLogger(
                logs_root_dir,
                version=most_recent_version,
            ),
            LoadableCSVLogger(
                logs_root_dir, version=most_recent_version, flush_logs_every_n_steps=100
            ),
        ],
        callbacks=[
            pl.callbacks.LearningRateMonitor(),
            ModelCheckpoint(save_last=True, save_top_k=0),
        ],
        max_steps=iterations,
        num_sanity_val_steps=1,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        default_root_dir=logs_root_dir,
        accumulate_grad_batches=args.batch_size_mult,
        enable_progress_bar=sys.stdout.isatty() or args.enable_progress,
        # gradient_clip_val=1.0,
        **strategy_kwargs,
        **check_val_opts,
    )

    valid_dataloaders = [
        DataLoader(
            PaddingDataset(
                ReorderSupportsByDistanceDataset(
                    MapDataset(
                        MapDataset(
                            Subset(
                                demonstrations,
                                np.random.permutation(len(demonstrations))[
                                    : args.limit_val_size
                                ],
                            ),
                            lambda x: (x[2], x[3], x[0], x[1], x[4], x[5], x[6]),
                        ),
                        lambda x: (
                            x[0],
                            [x[1]] * len(x[-1])
                            if not isinstance(x[1][0], list)
                            else x[1],
                            x[2],
                            x[3],
                            x[4],
                            x[5],
                            x[6],
                        ),
                    ),
                    args.metalearn_demonstrations_limit,
                ),
                (
                    (args.pad_state_to, None),
                    (args.metalearn_demonstrations_limit, args.pad_state_to, None),
                    32,
                    128,
                    (args.metalearn_demonstrations_limit, 32),
                    (args.metalearn_demonstrations_limit, 128),
                ),
                (pad_state, pad_state, pad_word, pad_action, pad_word, pad_action),
            ),
            pin_memory=True,
        )
        for demonstrations in meta_valid_demonstrations_dict.values()
    ]

    meta_trainer.fit(meta_module, train_dataloader, valid_dataloaders, ckpt_path="last")
    print(f"Done, saving {model_path}")
    meta_trainer.save_checkpoint(f"{model_path}")


if __name__ == "__main__":
    main()
