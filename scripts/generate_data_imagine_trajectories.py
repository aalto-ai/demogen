import argparse
import itertools
import os
import math
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from positional_encodings.torch_encodings import PositionalEncoding1D
from collections import defaultdict

from gscan_metaseq2seq.models.embedding import BOWEmbedding
from gscan_metaseq2seq.util.dataset import (
    PaddingDataset,
    ReshuffleOnIndexZeroDataset,
    MapDataset,
)
from gscan_metaseq2seq.util.load_data import load_data
from gscan_metaseq2seq.util.logging import LoadableCSVLogger
from gscan_metaseq2seq.util.scheduler import transformer_optimizer_config

from tqdm.auto import tqdm


class MonotonicRandomPositionEmbedding(nn.Module):
    def __init__(self, num_positions, emb_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_positions, emb_dim)
        self.num_positions = num_positions

    def forward(self, x):
        permutation = (
            torch.from_numpy(
                np.sort(np.random.permutation(self.num_positions)[: x.shape[1]])
            )[None]
            .expand(x.shape[0], -1)
            .to(x.device)
        )

        return self.embedding(permutation)


class MaskedLanguageModel(pl.LightningModule):
    def __init__(
        self,
        num_words,
        pad_word_idx,
        eos_word_idx,
        lr=0.0001,
        wd=1e-2,
        emb_dim=128,
        nlayers=8,
        nhead=4,
        dropout=0.1,
        norm_first=False,
        decay_power=-1,
        warmup_proportion=0.14,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.embedding_instructions = nn.Embedding(num_words, emb_dim)
        self.positional_encoding = MonotonicRandomPositionEmbedding(16, emb_dim)
        self.transformer_encoder_instruction = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=emb_dim,
                dim_feedforward=emb_dim * 4,
                dropout=dropout,
                nhead=nhead,
                norm_first=norm_first,
            ),
            num_layers=nlayers,
        )
        self.project = nn.Linear(emb_dim, num_words)
        self.pad_word_idx = pad_word_idx
        self.eos_word_idx = eos_word_idx

    def configure_optimizers(self):
        return transformer_optimizer_config(
            self,
            self.hparams.lr,
            weight_decay=self.hparams.wd,
            decay_power=self.hparams.decay_power,
            warmup_proportion=self.hparams.warmup_proportion,
        )

    def forward(self, base_instruction, instruction_mask):
        masked_instruction = base_instruction.clone()
        masked_instruction[instruction_mask] = self.pad_word_idx

        encoded_instruction = self.embedding_instructions(
            masked_instruction
        ) + self.positional_encoding(masked_instruction)
        decoded_instruction = self.transformer_encoder_instruction(
            encoded_instruction.transpose(0, 1)
        ).transpose(0, 1)

        return self.project(decoded_instruction)

    def training_step(self, x, idx):
        (instruction,) = x

        instruction_mask = (
            torch.stack(
                [
                    torch.randperm(instruction.shape[-1])
                    for _ in range(instruction.shape[0])
                ]
            )
            < (instruction.shape[-1] * np.random.uniform(0.0, 0.7))
        ).to(self.device)
        logits = self(instruction, instruction_mask)

        # We only consider the CE loss for the masked tokens
        # but not the unmasked ones
        target_instruction = instruction.clone()
        target_instruction[~instruction_mask] = self.eos_word_idx

        loss = F.cross_entropy(
            logits.flatten(0, -2),
            target_instruction.flatten(),
            ignore_index=self.eos_word_idx,
        )
        self.log("loss", loss)

        return loss

    def validation_step(self, x, idx, dl_idx):
        (instruction,) = x

        instruction_mask = (
            torch.stack(
                [
                    torch.randperm(instruction.shape[-1])
                    for _ in range(instruction.shape[0])
                ]
            )
            < (instruction.shape[-1] * 0.2)
        ).to(self.device)
        logits = self(instruction, instruction_mask)

        # We only consider the CE loss for the masked tokens
        # but not the unmasked ones
        target_instruction = instruction.clone()
        target_instruction[~instruction_mask] = self.eos_word_idx

        loss = F.cross_entropy(
            logits.flatten(0, -2),
            target_instruction.flatten(),
            ignore_index=self.eos_word_idx,
        )
        self.log("vloss", loss, prog_bar=True)

        return loss


def sample_from_model(model, instruction, sample_n, noise_level=0.2, device="cpu"):
    model.eval()
    model.to(device)

    with torch.inference_mode():
        expanded_instruction = (
            instruction[:, None].expand(-1, sample_n, -1).flatten(0, 1)
        )
        expanded_instruction = expanded_instruction.to(device)

        instruction_mask = (
            torch.stack(
                [
                    torch.randperm(expanded_instruction.shape[-1])
                    for _ in range(expanded_instruction.shape[0])
                ]
            )
            < (expanded_instruction.shape[-1] * noise_level)
        ).to(device)
        logits = model(expanded_instruction, instruction_mask)

        samples = torch.distributions.Categorical(logits=logits).sample()

        resampled = (
            samples * instruction_mask + expanded_instruction * ~instruction_mask
        )

        return (
            instruction.cpu(),
            resampled.view(instruction.shape[0], sample_n, -1).cpu(),
            instruction_mask.view(instruction.shape[0], sample_n, -1).cpu(),
        )


class InducingPointEncoder(nn.Module):
    def __init__(self, emb_dim, nhead, nlayers, norm_first=False, dropout=0.1):
        super().__init__()
        self.inducing_point = nn.Parameter(torch.randn(emb_dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=emb_dim,
                dim_feedforward=emb_dim * 4,
                nhead=nhead,
                norm_first=norm_first,
                dropout=dropout,
            ),
            num_layers=nlayers,
        )

    def forward(self, x, key_padding_mask):
        return self.transformer(
            torch.cat(
                [x, self.inducing_point[None, None].expand(x.shape[0], 1, -1)], dim=1
            ).transpose(0, 1),
            src_key_padding_mask=torch.cat(
                [key_padding_mask, torch.zeros_like(key_padding_mask[:, :1])], dim=-1
            ),
        )[-1]


class InstructionCLIPBCE(pl.LightningModule):
    def __init__(
        self,
        num_words,
        pad_word_idx,
        lr=0.0001,
        wd=1e-2,
        emb_dim=128,
        nlayers=8,
        nhead=4,
        dropout=0.1,
        norm_first=False,
        decay_power=-1,
        warmup_proportion=0.14,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.embedding_instructions = nn.Embedding(num_words, emb_dim)
        self.positional_encoding = PositionalEncoding1D(emb_dim)
        self.state_encoder = BOWEmbedding(64, 7, emb_dim)
        self.state_encoder_projection = nn.Linear(7 * emb_dim, emb_dim)
        self.transformer_encoder_instruction = InducingPointEncoder(
            emb_dim, nhead, nlayers, norm_first=norm_first, dropout=dropout
        )
        self.transformer_encoder_state = InducingPointEncoder(
            emb_dim, nhead, nlayers, norm_first=norm_first, dropout=dropout
        )
        self.pad_word_idx = pad_word_idx

    def configure_optimizers(self):
        return transformer_optimizer_config(
            self,
            self.hparams.lr,
            weight_decay=self.hparams.wd,
            decay_power=self.hparams.decay_power,
            warmup_proportion=self.hparams.warmup_proportion,
        )

    def forward(self, x):
        instruction, state, instruction_padding, state_padding = x

        encoded_state = self.state_encoder(state)
        projected_state = self.state_encoder_projection(encoded_state)
        encoded_instruction = self.embedding_instructions(instruction)
        encoded_instruction = encoded_instruction + self.positional_encoding(
            encoded_instruction
        )

        return (
            self.transformer_encoder_instruction(
                encoded_instruction, instruction_padding
            )
            @ self.transformer_encoder_state(projected_state, state_padding).T
        )

    def training_step(self, x, idx):
        instruction, state = x

        instruction_pad = instruction == self.pad_word_idx
        state_pad = torch.zeros_like(state[..., 0])
        # Lets do input dropout by randomly masking parts of the state
        state_pad[
            :,
            torch.randperm(state_pad.shape[1], device=state_pad.device)
            < (state_pad.shape[1] * 0.2),
        ] = 1
        state_pad = state_pad.to(torch.bool)

        outer_product = self.forward((instruction, state, instruction_pad, state_pad))

        matching_instruction = (
            (
                instruction[None, :].expand(instruction.shape[0], -1, -1)
                == instruction[:, None].expand(-1, instruction.shape[0], -1)
            )
            .all(dim=-1)
            .float()
        )

        loss = F.binary_cross_entropy_with_logits(
            outer_product,
            matching_instruction,
            pos_weight=torch.tensor(
                matching_instruction.flatten().shape[0]
                / (matching_instruction[matching_instruction == 1].shape[0]),
                device=self.device,
                dtype=torch.float,
            ),
        )
        self.log("loss", loss)

        return loss

    def predict_step(self, x, idx):
        instruction, state = x

        instruction_pad = instruction == self.pad_word_idx
        state_pad = torch.zeros_like(state[..., 0], dtype=torch.bool)

        outer_product = self.forward((instruction, state, instruction_pad, state_pad))

        return torch.diag(outer_product)


def batched(iterable, n):
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, n))

        if not batch:
            break

        yield batch


class StateEncoderTransformer(nn.Module):
    def __init__(
        self,
        n_state_components,
        input_size,
        embedding_dim,
        nlayers,
        nhead,
        dropout_p,
        norm_first,
        pad_word_idx,
    ):
        super().__init__()
        self.n_state_components = n_state_components
        self.embedding_dim = embedding_dim
        self.state_embedding = BOWEmbedding(
            64, self.n_state_components, self.embedding_dim
        )
        self.state_projection = nn.Linear(
            self.n_state_components * self.embedding_dim, self.embedding_dim
        )
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.embedding_projection = nn.Linear(embedding_dim * 2, embedding_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.encoding = nn.Parameter(torch.randn(embedding_dim))
        self.pos_encoding = PositionalEncoding1D(embedding_dim)
        self.pad_word_idx = pad_word_idx
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=nhead,
                dim_feedforward=embedding_dim * 4,
                dropout=dropout_p,
                norm_first=norm_first,
            ),
            num_layers=nlayers,
        )

    def forward(self, state_padded, z_padded):
        state_padding_bits = torch.zeros_like(state_padded[..., 0]).bool()
        z_padding_bits = z_padded == self.pad_word_idx

        state_embed_seq = self.state_projection(self.state_embedding(state_padded))

        z_embed_seq = self.embedding(z_padded)
        z_embed_seq = torch.cat([self.pos_encoding(z_embed_seq), z_embed_seq], dim=-1)
        z_embed_seq = self.embedding_projection(z_embed_seq)
        state_embed_seq = self.dropout(state_embed_seq)
        z_embed_seq = self.dropout(z_embed_seq)

        z_embed_seq = torch.cat([state_embed_seq, z_embed_seq], dim=1)
        padding_bits = torch.cat([state_padding_bits, z_padding_bits], dim=-1)

        encoded_seq = self.transformer_encoder(
            z_embed_seq.transpose(1, 0),
            src_key_padding_mask=padding_bits,
        )

        # bs x emb_dim, z_seq_len x bs x emb_dim
        return encoded_seq, padding_bits


class DecoderTransformer(nn.Module):
    def __init__(
        self,
        n_state_components,
        hidden_size,
        output_size,
        nlayers,
        nhead,
        pad_action_idx,
        dropout_p=0.1,
        norm_first=False,
    ):
        #
        # Input
        #  hidden_size : number of hidden units in Transformer, and embedding size for output symbols
        #  output_size : number of output symbols
        #  nlayers : number of hidden layers
        #  dropout_p : dropout applied to symbol embeddings and Transformers
        #
        super().__init__()
        self.n_state_components = n_state_components
        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.tanh = nn.Tanh()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_projection = nn.Linear(hidden_size * 2, hidden_size)
        self.pos_encoding = PositionalEncoding1D(hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.pad_action_idx = pad_action_idx
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_size,
                dim_feedforward=hidden_size * 4,
                dropout=dropout_p,
                nhead=nhead,
                norm_first=norm_first,
            ),
            num_layers=nlayers,
        )
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, encoder_outputs, encoder_padding):
        # Run batch decoder forward for a single time step.
        #
        # Input
        #  input: LongTensor of length batch_size x seq_len (left-shifted targets)
        #  memory: encoder state
        #
        # Output
        #   output : unnormalized output probabilities, batch_size x output_size
        #
        # Embed each input symbol
        input_padding_bits = inputs == self.pad_action_idx

        embedding = self.embedding(inputs)  # batch_size x hidden_size
        embedding = self.embedding_projection(
            torch.cat([embedding, self.pos_encoding(embedding)], dim=-1)
        )
        embedding = self.dropout(embedding)

        decoded = self.decoder(
            tgt=embedding.transpose(0, 1),
            memory=encoder_outputs,
            memory_key_padding_mask=encoder_padding,
            tgt_key_padding_mask=input_padding_bits,
            tgt_mask=torch.triu(
                torch.full((inputs.shape[-1], inputs.shape[-1]), float("-inf")),
                diagonal=1,
            ).to(inputs.device),
        ).transpose(0, 1)

        return self.out(decoded)


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


def transformer_predict(transformer_learner, state, instruction, decode_len):
    state = state.to(transformer_learner.device)
    instruction = instruction.to(transformer_learner.device)
    encodings, key_padding_mask = transformer_learner.encode(state, instruction)

    # Recursive decoding, start with a batch of SOS tokens
    decoder_in = torch.tensor(
        transformer_learner.sos_action_idx,
        dtype=torch.long,
        device=transformer_learner.device,
    )[None].expand(instruction.shape[0], 1)

    logits = []

    with torch.inference_mode():
        for i in range(decode_len):
            logits.append(
                transformer_learner.decode_autoregressive(
                    decoder_in, encodings, key_padding_mask
                )[:, -1]
            )
            decoder_out = logits[-1].argmax(dim=-1)
            decoder_in = torch.cat([decoder_in, decoder_out[:, None]], dim=1)

        decoded = decoder_in
        # these are shifted off by one
        decoded_eq_mask = (
            (decoded == transformer_learner.eos_action_idx)
            .int()
            .cumsum(dim=-1)
            .bool()[:, :-1]
        )
        decoded = decoded[:, 1:]
        decoded[decoded_eq_mask] = transformer_learner.pad_action_idx
        logits = torch.stack(logits, dim=1)

    return decoded, logits


class TransformerLearner(pl.LightningModule):
    def __init__(
        self,
        n_state_components,
        x_categories,
        y_categories,
        embed_dim,
        dropout_p,
        nlayers,
        nhead,
        pad_word_idx,
        pad_action_idx,
        sos_action_idx,
        eos_action_idx,
        norm_first=False,
        lr=1e-4,
        wd=1e-2,
        warmup_proportion=0.001,
        decay_power=-1,
        predict_steps=64,
    ):
        super().__init__()
        self.encoder = StateEncoderTransformer(
            n_state_components,
            x_categories,
            embed_dim,
            nlayers,
            nhead,
            dropout_p,
            norm_first,
            pad_word_idx,
        )
        self.decoder = DecoderTransformer(
            n_state_components,
            embed_dim,
            y_categories,
            nlayers,
            nhead,
            pad_action_idx,
            dropout_p=dropout_p,
            norm_first=norm_first,
        )
        self.y_categories = y_categories
        self.pad_word_idx = pad_word_idx
        self.pad_action_idx = pad_action_idx
        self.sos_action_idx = sos_action_idx
        self.eos_action_idx = eos_action_idx

        self.apply(init_parameters)
        self.save_hyperparameters()

    def configure_optimizers(self):
        return transformer_optimizer_config(
            self,
            self.hparams.lr,
            warmup_proportion=self.hparams.warmup_proportion,
            weight_decay=self.hparams.wd,
            decay_power=self.hparams.decay_power,
        )

    def encode(self, states, queries):
        return self.encoder(states, queries)

    def decode_autoregressive(self, decoder_in, encoder_outputs, encoder_padding):
        return self.decoder(decoder_in, encoder_outputs, encoder_padding)

    def forward(self, states, queries, decoder_in):
        encoded, encoder_padding = self.encoder(states, queries)
        return self.decode_autoregressive(decoder_in, encoded, encoder_padding)

    def training_step(self, x, idx):
        query, targets, state = x
        actions_mask = targets == self.pad_action_idx

        decoder_in = torch.cat(
            [torch.ones_like(targets)[:, :1] * self.sos_action_idx, targets], dim=-1
        )

        # Now do the training
        preds = self.forward(state, query, decoder_in)[:, :-1]

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

    def validation_step(self, x, idx, dl_idx=0):
        query, targets, state = x
        actions_mask = targets == self.pad_action_idx

        decoder_in = torch.cat(
            [torch.ones_like(targets)[:, :1] * self.sos_action_idx, targets], dim=-1
        )

        # Now do the training
        preds = self.forward(state, query, decoder_in)[:, :-1]

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

    def predict_step(self, x, idx, dl_idx=0):
        instruction, target, state = x[:3]

        encodings, key_padding_mask = self.encode(state, instruction)

        # Recursive decoding, start with a batch of SOS tokens
        decoder_in = torch.tensor(
            self.sos_action_idx, dtype=torch.long, device=self.device
        )[None].expand(instruction.shape[0], 1)

        logits = []

        with torch.no_grad():
            for i in range(target.shape[1]):
                logits.append(
                    self.decode_autoregressive(decoder_in, encodings, key_padding_mask)[
                        :, -1
                    ]
                )
                decoder_out = logits[-1].argmax(dim=-1)
                decoder_in = torch.cat([decoder_in, decoder_out[:, None]], dim=1)

            decoded = decoder_in
            # these are shifted off by one
            decoded_eq_mask = (
                (decoded == self.eos_action_idx).int().cumsum(dim=-1).bool()[:, :-1]
            )
            decoded = decoded[:, 1:]
            decoded[decoded_eq_mask] = -1
            logits = torch.stack(logits, dim=1)

        exacts = (decoded == target).all(dim=-1).cpu().numpy()

        decoded = decoded.cpu().numpy()
        decoded_select_mask = decoded != -1
        decoded = [d[m] for d, m in zip(decoded, decoded_select_mask)]

        target = target.cpu().numpy()
        target = [d[d != -1] for d in target]

        instruction = instruction.cpu().numpy()
        instruction = [i[i != -1] for i in instruction]

        logits = logits.cpu().numpy()
        logits = [l[m] for l, m in zip(logits, decoded_select_mask)]

        return tuple([state, instruction, decoded, logits, exacts, target] + x[3:])


def generate_instructions_and_rank(
    instruction_gen_model,
    instruction_clip_model,
    transformer_prediction_model,
    dataloader,
    sample_n,
    batch_size,
    noise_level,
    decode_len,
    pad_word_idx,
    device="cpu",
):
    instruction_gen_model.to(device)
    instruction_gen_model.eval()
    instruction_clip_model.to(device)
    instruction_clip_model.eval()
    transformer_prediction_model.to(device)
    transformer_prediction_model.eval()

    instruction_clip_model.positional_encoding.cached_penc = None

    for batch in dataloader:
        instruction, targets, state = batch

        result_instrs, result_samples, result_samples_mask = sample_from_model(
            instruction_gen_model,
            instruction,
            sample_n,
            noise_level=noise_level,
            device=device,
        )

        # Now for each element in the batch, we take
        # the set (involves a conversion back to tuples
        # but its fine)
        result_sets = [
            set([tuple(x.tolist()) for x in result_sample])
            for result_sample in result_samples
        ]
        result_set_ids = np.concatenate(
            [np.array([i] * len(s)) for i, s in enumerate(result_sets)]
        )
        result_set_states = np.concatenate(
            [
                np.repeat(s[None], len(result_set), axis=0)
                for s, result_set in zip(state, result_sets)
            ]
        )

        result_sets_concat = np.concatenate(
            [
                np.stack([np.array(x) for x in result_sample_set])
                for result_sample_set in result_sets
            ]
        )

        per_id_results = defaultdict(list)

        # Now we take the result sets and we split into batches
        # of size 128 each and use that with the CLIP model
        for i in range(result_sets_concat.shape[0] // batch_size + 1):
            result_set_ids_batch = result_set_ids[i * batch_size : (i + 1) * batch_size]
            result_set_states_batch = result_set_states[
                i * batch_size : (i + 1) * batch_size
            ]
            result_set_batch = result_sets_concat[i * batch_size : (i + 1) * batch_size]

            result_set_ids_batch = torch.from_numpy(result_set_ids_batch).to(device)
            result_set_states_batch = torch.from_numpy(result_set_states_batch).to(
                device
            )
            result_set_batch = torch.from_numpy(result_set_batch).to(device)

            instruction_pad = result_set_batch == pad_word_idx
            state_pad = torch.zeros_like(result_set_states_batch[..., 0])
            state_pad = state_pad.to(torch.bool)

            with torch.inference_mode():
                encoded_state = instruction_clip_model.state_encoder(
                    result_set_states_batch
                )
                projected_state = instruction_clip_model.state_encoder_projection(
                    encoded_state
                )
                encoded_instruction = instruction_clip_model.embedding_instructions(
                    result_set_batch
                )
                encoded_instruction = (
                    encoded_instruction
                    + instruction_clip_model.positional_encoding(encoded_instruction)
                )

                decoded_instruction = (
                    instruction_clip_model.transformer_encoder_instruction(
                        encoded_instruction, instruction_pad
                    )
                )
                decoded_state = instruction_clip_model.transformer_encoder_state(
                    projected_state, state_pad
                )

                # Take the componentwise product
                scores = (decoded_instruction * decoded_state).sum(dim=-1)

            predicted_targets, logits = transformer_predict(
                transformer_prediction_model,
                result_set_states_batch,
                result_set_batch,
                decode_len,
            )

            # Now we populate per_id_results with the score
            # and the result_set_ids
            for batch_id, sample_result, predicted_target, score in zip(
                result_set_ids_batch.cpu(),
                result_set_batch.cpu(),
                predicted_targets.cpu(),
                scores.cpu(),
            ):
                batch_id = batch_id.item()
                if not (sample_result == instruction[batch_id]).all():
                    per_id_results[batch_id].append(
                        (sample_result.numpy(), predicted_target.numpy(), score.item())
                    )

        # Now we sort the per_id_results ascending by score
        per_id_results = {
            i: sorted(results, key=lambda x: -x[-1])
            for i, results in per_id_results.items()
        }

        # Now yield per_id, the state, the query instruction and all supports
        # and their scores
        for batch_id, sample_scores in sorted(
            per_id_results.items(), key=lambda x: x[0]
        ):
            yield (
                instruction[batch_id].numpy(),
                targets[batch_id].numpy(),
                state[batch_id].numpy(),
                state[batch_id].numpy(),
                [s[0] for s in sample_scores],
                [s[1] for s in sample_scores],
                [s[2] for s in sample_scores],
            )


def train_mlm(
    balanced_training_data,
    valid_demonstrations_dict,
    seed,
    mlm_iterations,
    pad_word,
    sos_word,
    vocab_size,
    device="cpu",
):
    dataset = ReshuffleOnIndexZeroDataset(
        PaddingDataset(
            MapDataset(balanced_training_data, lambda x: (x[0],)), (8,), (pad_word,)
        )
    )

    nlayers = 4
    nhead = 8
    hidden_size = 128
    dropout_p = 0.1
    train_batch_size = 128
    batch_size_mult = 1
    dataset_name = "gscan_normal"
    check_val_every = 500

    exp_name = "gscan_mlm"
    model_name = f"transformer_l_{nlayers}_h_{nhead}_d_{hidden_size}"
    dataset_name = dataset_name
    effective_batch_size = train_batch_size * batch_size_mult
    exp_name = f"{exp_name}_s_{seed}_m_{model_name}_it_{mlm_iterations}_b_{effective_batch_size}_d_{dataset_name}_drop_{dropout_p}"
    model_dir = f"models/{exp_name}/{model_name}"
    model_path = f"{model_dir}/{exp_name}.pt"
    print(model_path)
    print(
        f"Batch size {train_batch_size}, mult {batch_size_mult}, total {train_batch_size * batch_size_mult}"
    )

    train_dataloader = DataLoader(
        dataset, batch_size=train_batch_size, pin_memory=True, shuffle=True
    )

    valid_dataloaders = [
        DataLoader(
            Subset(
                PaddingDataset(MapDataset(data, lambda x: (x[0],)), (8,), (pad_word,)),
                np.random.permutation(len(data))[:512],
            ),
            batch_size=train_batch_size,
            pin_memory=True,
        )
        for data in valid_demonstrations_dict.values()
    ]

    check_val_opts = {}
    interval = check_val_every / len(train_dataloader)

    # Every check_val_interval steps, regardless of how large the training dataloader is
    if interval > 1.0:
        check_val_opts["check_val_every_n_epoch"] = math.floor(interval)
    else:
        check_val_opts["val_check_interval"] = interval

    logs_root_dir = f"logs/{exp_name}/{model_name}/{dataset_name}/{seed}"

    model = MaskedLanguageModel(
        vocab_size,
        pad_word,
        sos_word,
        nlayers=nlayers,
        nhead=nhead,
        emb_dim=hidden_size,
        dropout=dropout_p,
        norm_first=True,
        lr=1e-4,
        decay_power=-1,
        warmup_proportion=0.1,
    )

    trainer = pl.Trainer(
        logger=[
            TensorBoardLogger(logs_root_dir),
            LoadableCSVLogger(logs_root_dir, flush_logs_every_n_steps=10),
        ],
        callbacks=[pl.callbacks.LearningRateMonitor()],
        max_steps=mlm_iterations,
        num_sanity_val_steps=10,
        gpus=1 if device == "cuda" else 0,
        precision=16 if device == "cuda" else 32,
        default_root_dir=logs_root_dir,
        accumulate_grad_batches=batch_size_mult,
        gradient_clip_val=0.2,
        **check_val_opts,
    )

    trainer.fit(model, train_dataloader, valid_dataloaders)

    return model


def train_clip(
    balanced_training_data,
    valid_demonstrations_dict,
    seed,
    clip_iterations,
    pad_word,
    vocab_size,
    device="cpu",
):
    nlayers = 4
    nhead = 8
    hidden_size = 128
    dropout_p = 0.1
    train_batch_size = 128
    batch_size_mult = 1
    dataset_name = "gscan_normal"
    check_val_every = 500

    clip_train_dataloader = DataLoader(
        ReshuffleOnIndexZeroDataset(
            PaddingDataset(
                MapDataset(balanced_training_data, lambda x: (x[0], x[2])),
                (8, None),
                (pad_word, None),
            )
        ),
        batch_size=train_batch_size,
        pin_memory=True,
        shuffle=True,
    )

    clip_valid_dataloaders = [
        DataLoader(
            PaddingDataset(
                MapDataset(data, lambda x: (x[0], x[2])), (8, None), (pad_word, None)
            ),
            batch_size=16,
            pin_memory=True,
        )
        for data in valid_demonstrations_dict.values()
    ]

    pl.seed_everything(seed)
    instruction_clip = InstructionCLIPBCE(
        vocab_size,
        pad_word,
        nlayers=4,
        nhead=8,
        emb_dim=512,
        dropout=dropout_p,
        norm_first=False,
        lr=1e-4,
        decay_power=-1,
        warmup_proportion=0.1,
    )

    exp_name = "clip"
    model_name = "transformer"
    dataset_name = "gscan_normal"

    logs_root_dir = f"logs/{exp_name}/{model_name}/{dataset_name}/{seed}"

    check_val_opts = {}
    interval = check_val_every / len(clip_train_dataloader)

    # Every check_val_interval steps, regardless of how large the training dataloader is
    if interval > 1.0:
        check_val_opts["check_val_every_n_epoch"] = math.floor(interval)
    else:
        check_val_opts["val_check_interval"] = interval

    instruction_clip_trainer = pl.Trainer(
        logger=[
            TensorBoardLogger(logs_root_dir),
            LoadableCSVLogger(logs_root_dir, flush_logs_every_n_steps=10),
        ],
        callbacks=[pl.callbacks.LearningRateMonitor()],
        max_steps=clip_iterations,
        num_sanity_val_steps=10,
        gpus=1 if device == "cuda" else 0,
        precision=16 if device == "cuda" else 32,
        default_root_dir=logs_root_dir,
        accumulate_grad_batches=batch_size_mult,
        gradient_clip_val=0.2,
        **check_val_opts,
    )

    instruction_clip_trainer.fit(
        instruction_clip, clip_train_dataloader, clip_valid_dataloaders
    )

    return instruction_clip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data", type=str, required=True)
    parser.add_argument("--validation-data-directory", type=str, required=True)
    parser.add_argument("--dictionary", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mlm-train-iterations", type=int, default=100000)
    parser.add_argument("--clip-train-iterations", type=int, default=100000)
    parser.add_argument("--load-mlm-model", type=str)
    parser.add_argument("--save-mlm-model", type=str)
    parser.add_argument("--load-transformer-model", type=str, required=True)
    parser.add_argument("--load-clip-model", type=str)
    parser.add_argument("--save-mlm-model", type=str)
    parser.add_argument("--data-output-directory", type=str, required=True)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--only-splits", nargs="*", description="Which splits to include"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    seed = args.seed
    mlm_iterations = args.mlm_train_iterations
    clip_iterations = args.clip_train_iterations

    (
        (
            WORD2IDX,
            ACTION2IDX,
            color_dictionary,
            noun_dictionary,
        ),
        (train_demonstrations, valid_demonstrations_dict),
    ) = load_data(args.training_data, args.validation_data_directory, args.dictionary)

    pad_action = ACTION2IDX["[pad]"]
    pad_word = WORD2IDX["[pad]"]

    IDX2WORD = {i: w for w, i in WORD2IDX.items()}

    training_data_indices_by_command = {}
    for i in range(len(train_demonstrations)):
        if WORD2IDX["cautiously"] in train_demonstrations[i][0]:
            continue

        cmd = " ".join(map(lambda x: IDX2WORD[x], train_demonstrations[i][0]))
        if cmd not in training_data_indices_by_command:
            training_data_indices_by_command[cmd] = []
        training_data_indices_by_command[cmd].append(i)

    min_len = min(
        [(cmd, len(x)) for cmd, x in training_data_indices_by_command.items()],
        key=lambda x: x[-1],
    )[-1]
    balanced_training_data = list(
        itertools.chain.from_iterable(
            [
                [train_demonstrations[i] for i in x[:min_len]]
                for x in training_data_indices_by_command.values()
            ]
        )
    )

    model = train_mlm(
        balanced_training_data,
        valid_demonstrations_dict,
        seed,
        0 if args.load_mlm else mlm_iterations,
        pad_word,
        WORD2IDX["[sos]"],
        len(WORD2IDX),
        device=args.device,
    )

    if args.load_mlm:
        model.load_state_dict(torch.load(args.load_mlm))

    if args.save_mlm:
        torch.save(model.state_dict(), args.save_mlm)

    instruction_clip = train_clip(
        balanced_training_data,
        valid_demonstrations_dict,
        seed,
        0 if args.load_clip else clip_iterations,
        pad_word,
        len(WORD2IDX),
        device=args.device,
    )

    if args.load_clip:
        instruction_clip.load_state_dict(torch.load(args.load_clip))

    if args.save_clip:
        torch.save(instruction_clip.state_dict(), args.save_clip)

    instruction_clip.positional_encoding.cached_penc = None

    os.makedirs(os.path.join(args.data_output_directory), exist_ok=True)

    transformer_model_weights = torch.load(args.load_transformer_model)
    transformer_model = TransformerLearner(
        **transformer_model_weights["hyper_parameters"]
    )
    transformer_model.load_state_dict(transformer_model_weights["state_dict"])

    transformer_model_trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        precision=16 if torch.cuda.is_available() else None,
    )
    transformer_model_trainer.validate(
        transformer_model,
        [
            DataLoader(
                Subset(
                    PaddingDataset(data, (8, 128, None), (pad_word, pad_action, None)),
                    np.random.permutation(512),
                ),
                batch_size=16,
                pin_memory=True,
            )
            for data in valid_demonstrations_dict.values()
        ],
    )

    dataloader_splits = {
        split: DataLoader(
            PaddingDataset(demos, (8, 128, None), (pad_word, pad_action, None)),
            batch_size=16,
            pin_memory=True,
        )
        for split, demos in zip(
            itertools.chain.from_iterable([valid_demonstrations_dict.keys(), "train"]),
            itertools.chain.from_iterable(
                [valid_demonstrations_dict.values(), train_demonstrations]
            ),
        )
        if not args.only_splits or split in args.only_splits
    }

    for split, dataloader in tqdm(dataloader_splits):
        os.makedirs(os.path.join(args.data_output_directory, split), exist_ok=True)

        for i, batch in enumerate(
            batched(
                generate_instructions_and_rank(
                    model,
                    instruction_clip,
                    transformer_model,
                    tqdm(dataloader),
                    256,
                    batch_size=args.batch_size,
                    noise_level=0.1,
                    decode_len=128,
                    pad_word_idx=pad_word,
                    device=args.device,
                ),
                1000,
            )
        ):
            with open(
                os.path.join(args.data_output_directory, split, f"{i}.pb"), "wb"
            ) as f:
                pickle.dump(batch, f)
