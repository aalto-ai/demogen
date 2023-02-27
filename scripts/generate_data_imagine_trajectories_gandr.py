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
from sklearn.preprocessing import StandardScaler
from pytorch_lightning.loggers import TensorBoardLogger
from positional_encodings.torch_encodings import PositionalEncoding1D

from gscan_metaseq2seq.models.embedding import BOWEmbedding
from gscan_metaseq2seq.util.dataset import (
    PaddingDataset,
    MapDataset,
)
from gscan_metaseq2seq.util.load_data import (
    load_data_directories,
    load_concat_pickle_files_from_directory,
)
from gscan_metaseq2seq.util.logging import LoadableCSVLogger
from gscan_metaseq2seq.util.scheduler import transformer_optimizer_config
from gscan_metaseq2seq.models.enc_dec_transformer.enc_dec_transformer_model import (
    TransformerLearner,
    autoregressive_model_unroll_predictions,
)

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


def train_transformer(
    weights_path,
    dataset,
    validation_datasets,
    seed,
    transformer_train_iterations,
    batch_size,
    x_categories,
    y_categories,
    pad_word_idx,
    pad_action_idx,
    sos_action_idx,
    eos_action_idx,
    hidden_size=128,
    nlayers=8,
    nhead=8,
    device="cpu",
):
    dropout_p = 0.0
    train_batch_size = batch_size
    batch_size_mult = 1
    dataset_name = "gscan"
    check_val_every = 8000

    exp_name = "transformer"
    model_name = f"transformer_l_{nlayers}_h_{nhead}_d_{hidden_size}"
    dataset_name = dataset_name
    effective_batch_size = train_batch_size * batch_size_mult
    exp_name = f"{exp_name}_s_{seed}_m_{model_name}_it_{transformer_train_iterations}_b_{effective_batch_size}_d_{dataset_name}_drop_{dropout_p}"
    model_dir = f"models/{exp_name}/{model_name}"
    model_path = f"{model_dir}/{exp_name}.pt"
    print(model_path)
    print(
        f"Batch size {train_batch_size}, mult {batch_size_mult}, total {train_batch_size * batch_size_mult}"
    )

    pl.seed_everything(seed)
    train_dataloader = DataLoader(
        dataset, batch_size=train_batch_size, pin_memory=True, shuffle=True
    )
    validation_dataloaders = [
        DataLoader(
            ds,
            batch_size=train_batch_size,
            pin_memory=True,
        )
        for ds in validation_datasets
    ]

    logs_root_dir = f"logs/{exp_name}/{model_name}/{dataset_name}/{seed}"

    model = TransformerLearner(
        7,
        x_categories,
        y_categories,
        embed_dim=hidden_size,
        dropout_p=dropout_p,
        nlayers=nlayers,
        nhead=nhead,
        pad_word_idx=pad_word_idx,
        pad_action_idx=pad_action_idx,
        sos_action_idx=sos_action_idx,
        eos_action_idx=eos_action_idx,
        wd=1e-2,
        lr=1e-4,
        decay_power=-1,
        warmup_proportion=0.1,
    )
    print(model)

    if weights_path:
        model.load_state_dict(torch.load(weights_path))

    pl.seed_everything(seed)
    trainer = pl.Trainer(
        logger=[
            TensorBoardLogger(logs_root_dir),
            LoadableCSVLogger(logs_root_dir, flush_logs_every_n_steps=10),
        ],
        callbacks=[pl.callbacks.LearningRateMonitor()],
        max_steps=transformer_train_iterations,
        num_sanity_val_steps=10,
        gpus=1 if device == "cuda" else 0,
        precision=16 if device == "cuda" else 32,
        default_root_dir=logs_root_dir,
        accumulate_grad_batches=batch_size_mult,
        gradient_clip_val=0.2,
    )

    trainer.fit(model, train_dataloader, validation_dataloaders)

    return model


def to_count_matrix(action_word_arrays, word_vocab_size, action_vocab_size):
    count_matrix = np.zeros(
        (len(action_word_arrays), word_vocab_size + action_vocab_size)
    )

    for i, (word_array, action_array) in enumerate(action_word_arrays):
        for element in word_array:
            count_matrix[i, element] += 1
        for element in action_array:
            count_matrix[i, element + word_vocab_size] += 1

    return count_matrix


def to_tfidf(tfidf_transformer, count_matrix):
    return tfidf_transformer.transform(count_matrix).todense().astype("float32")


def transformer_predict(transformer_learner, state, instruction, decode_len):
    state = state.to(transformer_learner.device)
    instruction = instruction.to(transformer_learner.device)
    dummy_targets = torch.zeros(
        instruction.shape[0],
        decode_len,
        dtype=torch.long,
        device=transformer_learner.device,
    )

    decoded, logits, exacts, _ = autoregressive_model_unroll_predictions(
        transformer_learner,
        (state, instruction),
        dummy_targets,
        transformer_learner.sos_action_idx,
        transformer_learner.eos_action_idx,
        transformer_learner.pad_action_idx,
    )

    return decoded, logits


def gandr_like_search(
    transformer_prediction_model,
    index,
    train_dataset,
    tfidf_transformer,
    scaler,
    state_autoencoder_transformer,
    dataloader,
    sample_n,
    decode_len,
    pad_word_idx,
    pad_action_idx,
    word_vocab_size,
    action_vocab_size,
    device="cpu",
):
    transformer_prediction_model.to(device)
    transformer_prediction_model.eval()
    state_autoencoder_transformer.to(device)
    state_autoencoder_transformer.eval()

    for batch in dataloader:
        instruction, targets, state = batch

        state_encodings = None
        if state_autoencoder_transformer is not None:
            with torch.inference_mode():
                state_encodings = (
                    state_autoencoder_transformer.encode_to_vector(state.to(device))
                    .detach()
                    .cpu()
                    .numpy()
                )

        predicted_targets, logits = transformer_predict(
            transformer_prediction_model,
            state,
            instruction,
            decode_len,
        )

        tfidf_vectors = to_tfidf(
            tfidf_transformer,
            to_count_matrix(
                [
                    (i[i != pad_word_idx], t[t != pad_action_idx])
                    for i, t in zip(instruction, predicted_targets)
                ],
                word_vocab_size,
                action_vocab_size,
            ),
        )
        unscaled_vectors = (
            np.concatenate([tfidf_vectors, state_encodings], axis=-1)
            if state_encodings is not None
            else tfidf_vectors
        )
        scaled_vectors = scaler.transform(unscaled_vectors)

        near_neighbour_distances_batch, near_neighbour_indices_batch = index.search(
            scaled_vectors,
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


class StateCLIPTransformer(pl.LightningModule):
    def __init__(
        self,
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
        self.pos_encoding = PositionalEncoding1D(emb_dim)
        self.embedding = BOWEmbedding(16, 7, emb_dim)
        self.projection = nn.Linear(emb_dim * 7, emb_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=emb_dim,
                dim_feedforward=emb_dim * 4,
                dropout=dropout,
                nhead=nhead,
                norm_first=norm_first,
            ),
            num_layers=nlayers,
        )
        self.latent = nn.Parameter(torch.randn(emb_dim))
        self.project = nn.Linear(emb_dim, 16 * 7)

    def configure_optimizers(self):
        return transformer_optimizer_config(
            self,
            self.hparams.lr,
            weight_decay=self.hparams.wd,
            decay_power=self.hparams.decay_power,
            warmup_proportion=self.hparams.warmup_proportion,
        )

    def encode_to_vector(self, state):
        mask = (state == 0).all(dim=-1)
        embedded_state = self.projection(self.embedding(state))
        latent_seq = self.latent[None, None].expand(embedded_state.shape[0], 1, -1)

        encoded_state_with_latent = self.transformer_encoder(
            torch.cat([embedded_state, latent_seq], dim=1).transpose(1, 0),
            src_key_padding_mask=torch.cat(
                [mask, torch.zeros_like(mask[:, :1])], dim=1
            ),
        ).transpose(0, 1)

        return encoded_state_with_latent[:, -1]

    def forward(self, state):
        mask = (state == 0).all(dim=-1)
        embedded_state = self.projection(self.embedding(state))
        latent_seq = self.latent[None, None].expand(embedded_state.shape[0], 1, -1)

        orig_key_padding_mask = torch.cat([mask, torch.zeros_like(mask[:, :1])], dim=1)
        dropout_key_padding_mask = (
            torch.rand_like(orig_key_padding_mask.float()) < 0.3
        ) * orig_key_padding_mask

        orig_encoded_state_with_latent = self.transformer_encoder(
            torch.cat([embedded_state, latent_seq], dim=1).transpose(1, 0),
            src_key_padding_mask=torch.cat(
                [mask, torch.zeros_like(mask[:, :1])], dim=1
            ),
        ).transpose(0, 1)

        dropout_encoded_state_with_latent = self.transformer_encoder(
            torch.cat([embedded_state, latent_seq], dim=1).transpose(1, 0),
            src_key_padding_mask=dropout_key_padding_mask,
        ).transpose(0, 1)

        return orig_encoded_state_with_latent[
            :, -1
        ] @ dropout_encoded_state_with_latent[:, -1].transpose(0, 1)

    def training_step(self, x, idx):
        (state,) = x

        # Split into the seven components, then flatten
        # into one big sequence
        outer_product = self.forward(state)

        label = torch.arange(state.shape[0], device=state.device, dtype=torch.long)
        loss = (
            F.cross_entropy(outer_product, label)
            + F.cross_entropy(outer_product.transpose(1, 0), label)
        ) / 2

        self.log("tloss", loss)

        return loss


def train_state_autoencoder(
    weights_path,
    dataset,
    seed,
    mlm_train_iterations,
    batch_size,
    hidden_size=128,
    nlayers=4,
    nhead=8,
    device="cpu",
):
    dropout_p = 0.1
    train_batch_size = batch_size
    batch_size_mult = 1
    dataset_name = "gscan"
    check_val_every = 8000

    exp_name = "state_autoencoder"
    model_name = f"transformer_l_{nlayers}_h_{nhead}_d_{hidden_size}"
    dataset_name = dataset_name
    effective_batch_size = train_batch_size * batch_size_mult
    exp_name = f"{exp_name}_s_{seed}_m_{model_name}_it_{mlm_train_iterations}_b_{effective_batch_size}_d_{dataset_name}_drop_{dropout_p}"
    model_dir = f"models/{exp_name}/{model_name}"
    model_path = f"{model_dir}/{exp_name}.pt"
    print(model_path)
    print(
        f"Batch size {train_batch_size}, mult {batch_size_mult}, total {train_batch_size * batch_size_mult}"
    )

    train_dataloader = DataLoader(
        dataset, batch_size=train_batch_size, pin_memory=True, shuffle=True
    )

    logs_root_dir = f"logs/{exp_name}/{model_name}/{dataset_name}/{seed}"

    model = StateCLIPTransformer(
        nlayers=nlayers,
        nhead=nhead,
        emb_dim=hidden_size,
        dropout=dropout_p,
        norm_first=True,
        lr=1e-4,
        decay_power=-1,
        warmup_proportion=0.1,
    )
    print(model)

    if weights_path:
        model.load_state_dict(torch.load(weights_path))

    trainer = pl.Trainer(
        logger=[
            TensorBoardLogger(logs_root_dir),
            LoadableCSVLogger(logs_root_dir, flush_logs_every_n_steps=10),
        ],
        callbacks=[pl.callbacks.LearningRateMonitor()],
        max_steps=mlm_train_iterations,
        num_sanity_val_steps=10,
        gpus=1 if device == "cuda" else 0,
        precision=16 if device == "cuda" else 32,
        default_root_dir=logs_root_dir,
        accumulate_grad_batches=batch_size_mult,
        gradient_clip_val=0.2,
    )

    trainer.fit(model, train_dataloader)

    return model


def load_state_encodings(directory):
    return {
        k: load_concat_pickle_files_from_directory(os.path.join(directory, k))
        for k in filter(
            lambda x: os.path.isdir(os.path.join(directory, x)), os.listdir(directory)
        )
    }


def save_state_encodings(state_encodings_dict, directory):
    for key, encodings in state_encodings_dict.items():
        path = os.path.join(directory, key)
        os.makedirs(path, exist_ok=True)

        for i, batch in enumerate(batched(encodings, 10000)):
            with open(os.path.join(path, f"{i}.pb"), "wb") as f:
                pickle.dump(np.stack(batch), f)


def generate_state_encodings(
    state_autoencoder_transformer,
    train_demonstrations,
    valid_demonstrations_dict,
    batch_size,
    device="cpu",
):
    dataloaders = {
        split: DataLoader(
            PaddingDataset(
                MapDataset(demos, lambda x: (x[-1],)),
                ((36, 7),),
                (0,),
            ),
            batch_size=batch_size,
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
    }

    state_autoencoder_transformer.to(device)
    state_autoencoder_transformer.eval()

    with torch.inference_mode():
        encoded_states = {
            split: np.concatenate(
                list(
                    map(
                        lambda x: state_autoencoder_transformer.encode_to_vector(
                            x[0].long().to(device)
                        )
                        .detach()
                        .cpu()
                        .numpy(),
                        tqdm(dl),
                    )
                )
            )
            for split, dl in tqdm(dataloaders.items())
        }

    return encoded_states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data", type=str, required=True)
    parser.add_argument("--dictionary", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-transformer-model", type=str)
    parser.add_argument("--load-transformer-model", type=str)
    parser.add_argument("--transformer-iterations", type=int, default=150000)
    parser.add_argument(
        "--state-autoencoder-transformer-iterations", type=int, default=50000
    )
    parser.add_argument("--load-state-autoencoder-transformer", type=str)
    parser.add_argument("--save-state-autoencoder-transformer", type=str)
    parser.add_argument("--load-state-encodings", type=str)
    parser.add_argument("--save-state-encodings", type=str)
    parser.add_argument("--data-output-directory", type=str, required=True)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--only-splits", nargs="*", help="Which splits to include")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--include-state", action="store_true")
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

    # Set to None by default at least so that we can handle it not being there
    state_autoencoder_transformer = None
    state_encodings_by_split = None

    if args.include_state:
        state_autoencoder_transformer = train_state_autoencoder(
            args.load_state_autoencoder_transformer,
            PaddingDataset(
                MapDataset(train_demonstrations, lambda x: (x[-1],)), ((36, 7),), (0,)
            ),
            args.seed,
            args.state_autoencoder_transformer_iterations
            if args.save_state_autoencoder_transformer
            else 0,
            args.batch_size,
            hidden_size=args.hidden_size,
            device=args.device,
        )

        if args.save_state_autoencoder_transformer:
            torch.save(
                state_autoencoder_transformer.state_dict(),
                args.save_state_autoencoder_transformer,
            )

    transformer_validation_datasets = [
        Subset(
            PaddingDataset(data, (8, 128, (36, 7)), (pad_word, pad_action, 0)),
            np.random.permutation(512),
        )
        for data in valid_demonstrations_dict.values()
    ]

    transformer_model = train_transformer(
        args.load_transformer_model,
        PaddingDataset(
            train_demonstrations,
            (8, 128, (36, 7)),
            (
                pad_word,
                pad_action,
                0,
            ),
        ),
        transformer_validation_datasets,
        args.seed,
        args.transformer_iterations if args.save_transformer_model else 0,
        args.batch_size,
        len(WORD2IDX),
        len(ACTION2IDX),
        pad_word,
        pad_action,
        ACTION2IDX["[sos]"],
        ACTION2IDX["[eos]"],
        hidden_size=128,
        nlayers=8,
        nhead=8,
        device=args.device,
    )

    if args.save_transformer_model:
        torch.save(transformer_model.state_dict(), args.save_transformer_model)

    transformer_model_trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        precision=16 if torch.cuda.is_available() else None,
    )

    transformer_model_trainer.validate(
        transformer_model,
        [
            DataLoader(
                ds,
                batch_size=16,
                pin_memory=True,
            )
            for ds in transformer_validation_datasets
        ],
    )

    print(args.offset, args.offset + (0 if args.limit is None else args.limit))

    if args.include_state:
        if args.load_state_encodings:
            state_encodings_by_split = load_state_encodings(args.load_state_encodings)
        else:
            state_encodings_by_split = generate_state_encodings(
                state_autoencoder_transformer,
                train_demonstrations,
                valid_demonstrations_dict,
                args.batch_size,
                device=args.device,
            )

        if args.save_state_encodings:
            save_state_encodings(state_encodings_by_split, args.save_state_encodings)

    # Make an index from the training data
    np.random.seed(args.seed)
    index = (
        faiss.IndexFlatIP(len(WORD2IDX) + len(ACTION2IDX) + args.hidden_size)
        if args.include_states
        else faiss.IndexFlatL2(len(WORD2IDX) + len(ACTION2IDX))
    )
    count_matrix = to_count_matrix(
        [
            (instruction, actions)
            for instruction, actions, state in train_demonstrations
        ],
        len(WORD2IDX),
        len(ACTION2IDX),
    )
    tfidf_transformer = TfidfTransformer()
    tfidf_transformer.fit(count_matrix)
    unscaled_vectors = (
        np.concatenate(
            [
                tfidf_transformer.transform(count_matrix).todense().astype("float32"),
                state_encodings_by_split["train"],
            ],
            axis=-1,
        )
        if state_encodings_by_split is not None
        else tfidf_transformer.transform(count_matrix).todense().astype("float32")
    )

    scaler = StandardScaler()
    scaler.fit(unscaled_vectors)
    scaled_vectors = scaler.transform(unscaled_vectors)

    index.add(scaled_vectors)

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
                (8, 128, (36, 7)),
                (pad_word, pad_action, 0),
            ),
            batch_size=args.batch_size,
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
                    scaler,
                    state_autoencoder_transformer,
                    tqdm(dataloader),
                    16,
                    decode_len=128,
                    pad_word_idx=pad_word,
                    pad_action_idx=pad_action,
                    word_vocab_size=len(WORD2IDX),
                    action_vocab_size=len(ACTION2IDX),
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