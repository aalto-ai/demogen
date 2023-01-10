import argparse
import os
import math
import torch
import numpy as np
import sys
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


from gscan_metaseq2seq.util.dataset import PaddingDataset, ReshuffleOnIndexZeroDataset
from gscan_metaseq2seq.util.load_data import load_data_directories
from gscan_metaseq2seq.util.logging import LoadableCSVLogger
from gscan_metaseq2seq.models.enc_dec_transformer.enc_dec_transformer_model import TransformerLearner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-demonstrations", type=str, required=True)
    parser.add_argument("--valid-demonstrations-directory", type=str, required=True)
    parser.add_argument("--dictionary", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--valid-batch-size", type=int, default=128)
    parser.add_argument("--batch-size-mult", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--nlayers", type=int, default=8)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dropout-p", type=float, default=0.0)
    parser.add_argument("--norm-first", action="store_true")
    parser.add_argument("--precision", type=int, choices=(16, 32), default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--warmup-proportion", type=float, default=0.1)
    parser.add_argument("--decay-power", type=int, default=-1)
    parser.add_argument("--iterations", type=int, default=2500000)
    parser.add_argument("--check-val-every", type=int, default=1000)
    parser.add_argument("--limit-val-size", type=int, default=None)
    parser.add_argument("--enable-progress", action="store_true")
    parser.add_argument("--restore-from-checkpoint", action="store_true")
    parser.add_argument("--version", type=int, default=None)
    parser.add_argument("--tag", type=str, default="none")
    parser.add_argument("--dataset-name", type=str, default="gscan")
    parser.add_argument("--pad-instructions-to", type=int, default=8)
    parser.add_argument("--pad-actions-to", type=int, default=128)
    parser.add_argument("--pad-state-to", type=int, default=36)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--dataloader-ncpus", type=int, default=1)
    args = parser.parse_args()

    exp_name = "gscan"
    model_name = f"transformer_encoder_only_decode_actions_l_{args.nlayers}_h_{args.nhead}_d_{args.hidden_size}"
    dataset_name = args.dataset_name
    effective_batch_size = args.train_batch_size * args.batch_size_mult
    exp_name = f"{exp_name}_s_{args.seed}_m_{model_name}_it_{args.iterations}_b_{effective_batch_size}_d_{dataset_name}_t_{args.tag}_drop_{args.dropout_p}"
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
    ) = load_data_directories(args.train_demonstrations, args.dictionary)

    IDX2WORD = {i: w for w, i in WORD2IDX.items()}
    IDX2ACTION = {i: w for w, i in ACTION2IDX.items()}

    pad_word = WORD2IDX["[pad]"]
    pad_action = ACTION2IDX["[pad]"]
    sos_action = ACTION2IDX["[sos]"]
    eos_action = ACTION2IDX["[eos]"]

    pl.seed_everything(0)
    train_dataset = ReshuffleOnIndexZeroDataset(
        PaddingDataset(
            train_demonstrations,
            (args.pad_instructions_to, args.pad_actions_to, (args.pad_state_to, 7)),
            (pad_word, pad_action, 0),
        )
    )

    pl.seed_everything(seed)
    meta_module = TransformerLearner(
        7,
        len(IDX2WORD),
        len(IDX2ACTION),
        args.hidden_size,
        args.dropout_p,
        args.nlayers,
        args.nhead,
        pad_word,
        pad_action,
        sos_action,
        eos_action,
        lr=args.lr,
        norm_first=args.norm_first,
        decay_power=args.decay_power,
        warmup_proportion=args.warmup_proportion,
    )
    print(meta_module)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        pin_memory=True,
        num_workers=args.dataloader_ncpus
    )

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

    logs_root_dir = f"{args.log_dir}/{exp_name}/{model_name}/{dataset_name}/{seed}"
    most_recent_version = args.version

    trainer = pl.Trainer(
        logger=[
            TensorBoardLogger(logs_root_dir, version=most_recent_version),
            LoadableCSVLogger(
                logs_root_dir, version=most_recent_version, flush_logs_every_n_steps=10
            ),
        ],
        callbacks=[pl.callbacks.LearningRateMonitor(), checkpoint_cb],
        max_steps=iterations,
        num_sanity_val_steps=10,
        gpus=1 if torch.cuda.is_available() else 0,
        precision=args.precision if torch.cuda.is_available() else 32,
        default_root_dir=logs_root_dir,
        accumulate_grad_batches=args.batch_size_mult,
        enable_progress_bar=sys.stdout.isatty() or args.enable_progress,
        gradient_clip_val=0.2,
        **check_val_opts,
    )

    trainer.fit(
        meta_module,
        train_dataloader,
        [
            DataLoader(
                PaddingDataset(
                    Subset(
                        demonstrations,
                        np.random.permutation(len(demonstrations))[
                            : args.limit_val_size
                        ],
                    ),
                    (
                        args.pad_instructions_to,
                        args.pad_actions_to,
                        (args.pad_state_to, 7),
                    ),
                    (pad_word, pad_action, 0),
                ),
                batch_size=max([args.train_batch_size, args.valid_batch_size]),
                pin_memory=True,
            )
            for demonstrations in valid_demonstrations_dict.values()
        ],
    )
    print(f"Done, saving {model_path}")
    trainer.save_checkpoint(f"{model_path}")


if __name__ == "__main__":
    main()
