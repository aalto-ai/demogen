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

import faiss
from sklearn.feature_extraction.text import TfidfTransformer
from pytorch_lightning.loggers import TensorBoardLogger
from positional_encodings.torch_encodings import PositionalEncoding1D
from collections import defaultdict

from gscan_metaseq2seq.models.embedding import BOWEmbedding
from gscan_metaseq2seq.util.dataset import (
    PaddingDataset,
    ReshuffleOnIndexZeroDataset,
    MapDataset,
)
from gscan_metaseq2seq.util.load_data import load_data_directories
from gscan_metaseq2seq.util.logging import LoadableCSVLogger
from gscan_metaseq2seq.util.scheduler import transformer_optimizer_config

from tqdm.auto import tqdm


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
        state_padding_bits = (state_padded == 0).all(dim=-1)
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


def to_count_matrix(actions_arrays, action_vocab_size):
    count_matrix = np.zeros((len(actions_arrays), action_vocab_size))

    for i, array in enumerate(actions_arrays):
        for element in array:
            count_matrix[i, element] += 1

    return count_matrix


def to_tfidf(tfidf_transformer, count_matrix):
    return tfidf_transformer.transform(count_matrix).todense().astype("float32")


def gandr_like_search(
    transformer_prediction_model,
    index,
    train_dataset,
    tfidf_transformer,
    dataloader,
    sample_n,
    decode_len,
    pad_action_idx,
    action_vocab_size,
    device="cpu",
):
    transformer_prediction_model.to(device)
    transformer_prediction_model.eval()

    for batch in dataloader:
        instruction, targets, state = batch

        predicted_targets, logits = transformer_predict(
            transformer_prediction_model,
            state,
            instruction,
            decode_len,
        )

        near_neighbour_distances_batch, near_neighbour_indices_batch = index.search(
            to_tfidf(
                tfidf_transformer,
                to_count_matrix(
                    [t[t != pad_action_idx] for t in predicted_targets],
                    action_vocab_size,
                ),
            ),
            sample_n,
        )

        near_neighbour_supports_batch = [
            (
                [train_dataset[i] for i in near_neighbour_indices],
                near_neighbour_distances,
            )
            for near_neighbour_indices, near_neighbour_distances in zip(
                near_neighbour_indices_batch, near_neighbour_distances_batch
            )
        ]

        for i, (supports, distances) in enumerate(near_neighbour_supports_batch):
            yield (
                instruction[i].numpy(),
                targets[i].numpy(),
                state[i].numpy(),
                [s[-1] for s in supports],  # support state
                [s[-3] for s in supports],  # support instruction
                [s[-2] for s in supports],  # support actions
                distances,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data", type=str, required=True)
    parser.add_argument("--validation-data-directory", type=str, required=True)
    parser.add_argument("--dictionary", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load-transformer-model", type=str, required=True)
    parser.add_argument("--data-output-directory", type=str, required=True)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--only-splits", nargs="*", help="Which splits to include")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    (
        (
            WORD2IDX,
            ACTION2IDX,
            color_dictionary,
            noun_dictionary,
        ),
        (train_demonstrations, valid_demonstrations_dict),
    ) = load_data_directories(args.training_data, args.dictionary)

    pad_action = ACTION2IDX["[pad]"]
    pad_word = WORD2IDX["[pad]"]

    IDX2WORD = {i: w for w, i in WORD2IDX.items()}

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

    print(args.offset, args.offset + args.limit)

    # Make an index from the training data
    index = faiss.IndexFlatL2(len(ACTION2IDX))
    count_matrix = to_count_matrix(
        [actions for instruction, actions, state in train_demonstrations],
        len(ACTION2IDX),
    )
    tfidf_transformer = TfidfTransformer()
    tfidf_transformer.fit(count_matrix)
    index.add(tfidf_transformer.transform(count_matrix).todense().astype("float32"))

    dataloader_splits = {
        split: DataLoader(
            PaddingDataset(
                Subset(
                    demos,
                    np.arange(
                        min(args.offset, len(demos)),
                        min(
                            args.offset
                            + (len(demos) if not args.limit else args.limit),
                            len(demos),
                        ),
                    ),
                ),
                (8, 128, None),
                (pad_word, pad_action, None),
            ),
            batch_size=16,
            pin_memory=True,
        )
        for split, demos in zip(
            itertools.chain.from_iterable(
                [valid_demonstrations_dict.keys(), ["train"]]
            ),
            itertools.chain.from_iterable(
                [valid_demonstrations_dict.values(), [train_demonstrations]]
            ),
        )
        if not args.only_splits or split in args.only_splits
    }

    for split, dataloader in tqdm(dataloader_splits.items()):
        os.makedirs(os.path.join(args.data_output_directory, split), exist_ok=True)

        for i, batch in enumerate(
            batched(
                gandr_like_search(
                    transformer_model,
                    index,
                    train_demonstrations,
                    tfidf_transformer,
                    tqdm(dataloader),
                    16,
                    decode_len=128,
                    action_vocab_size=len(ACTION2IDX),
                    pad_action_idx=pad_action,
                    device=args.device,
                ),
                1000,
            )
        ):
            with open(
                os.path.join(args.data_output_directory, split, f"{i}.pb"), "wb"
            ) as f:
                pickle.dump(batch, f)


if __name__ == "__main__":
    main()
