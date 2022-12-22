import argparse
import itertools
import os
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
from torch.utils.data import Dataset, DataLoader, Subset
from positional_encodings.torch_encodings import PositionalEncoding1D
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm.auto import trange

from gscan_metaseq2seq.models.embedding import BOWEmbedding
from gscan_metaseq2seq.util.dataset import PaddingDataset, ReshuffleOnIndexZeroDataset
from gscan_metaseq2seq.util.load_data import load_data
from gscan_metaseq2seq.util.logging import LoadableCSVLogger
from gscan_metaseq2seq.util.scheduler import transformer_optimizer_config


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def compute_schedule_variables(timesteps):
    # define beta schedule
    betas = linear_beta_schedule(timesteps=timesteps)

    # define alphas
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    return (
        betas,
        alphas,
        sqrt_recip_alphas,
        sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod,
        posterior_variance,
    )


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def add_q_noise(
    x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None
):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def p_sample(
    model,
    state,
    masked_instruction,
    x,
    t,
    t_index,
    betas,
    sqrt_one_minus_alphas_cumprod,
    sqrt_recip_alphas,
    posterior_variance,
):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    logits_state = model((masked_instruction, x, state, t), override_p_image_drop=0.0)
    logits_null = model((masked_instruction, x, state, t), override_p_image_drop=1.0)

    logits = logits_null + (logits_state - logits_null) * 2

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * logits / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise


def make_masked_instruction(
    base_instruction, pad_word_idx, mask_pct=0.3, contiguous_masking=False, device="cpu"
):
    instruction_mask = (
        torch.stack(
            [
                torch.randperm(base_instruction.shape[-1])
                for _ in range(base_instruction.shape[0])
            ]
        )
        < (base_instruction.shape[-1] * mask_pct)
    ).to(device)

    if contiguous_masking:
        mask_lowerbound = (
            torch.arange(base_instruction.shape[0], device=device)
            % (
                base_instruction.shape[-1]
                + math.floor(base_instruction.shape[1] * mask_pct)
            )
        ) - math.floor(base_instruction.shape[1] * mask_pct)
        mask_upperbound = mask_lowerbound + math.floor(
            base_instruction.shape[1] * mask_pct
        )
        instruction_mask_indices = torch.arange(
            base_instruction.shape[1], device=device
        )[None].expand(base_instruction.shape[0], -1)
        instruction_mask = torch.bitwise_and(
            instruction_mask_indices >= mask_lowerbound[:, None],
            instruction_mask_indices <= mask_upperbound[:, None],
        )

    masked_instruction = base_instruction.clone()
    masked_instruction[instruction_mask] = pad_word_idx

    return masked_instruction


def p_sample_loop(
    model,
    states,
    base_instruction,
    base_noisy_instructions,
    timesteps,
    betas,
    sqrt_one_minus_alphas_cumprod,
    sqrt_recip_alphas,
    posterior_variance,
    pad_word_idx,
    mask_pct=0.3,
    contiguous_masking=False,
):
    b = base_noisy_instructions.shape[0]
    # start from pure noise (for each example in the batch)
    imgs = []
    img = base_noisy_instructions

    masked_instruction = make_masked_instruction(
        base_instruction,
        pad_word_idx,
        mask_pct=mask_pct,
        contiguous_masking=contiguous_masking,
        device=model.device,
    )

    for i in reversed(trange(0, timesteps)):
        img = p_sample(
            model,
            states,
            masked_instruction,
            img,
            torch.full(
                (b,), i, device=base_noisy_instructions.device, dtype=torch.long
            ),
            i,
            betas,
            sqrt_one_minus_alphas_cumprod,
            sqrt_recip_alphas,
            posterior_variance,
        )
        imgs.append(img.cpu())
    return imgs


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class InstructionDiffusionModel(pl.LightningModule):
    def __init__(
        self,
        vocab_size,
        pad_word_idx,
        timesteps=200,
        lr=0.0001,
        wd=1e-2,
        nlayers=8,
        nhead=4,
        emb_dim=128,
        norm_first=False,
        dropout_p=0.0,
        decay_power=-2,
        warmup_proportion=0.1,
        instruction_samples=16,
        p_image_drop=0.1,
        mask_pct=0.3,
        contiguous_masking=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.batch_norm = nn.BatchNorm1d(vocab_size, affine=False)
        self.projection_instructions = nn.Linear(vocab_size, emb_dim)
        self.pe_instruction = PositionalEncoding1D(emb_dim)
        self.time_embedding = SinusoidalPositionEmbeddings(emb_dim)
        self.state_encoder = BOWEmbedding(64, 7, emb_dim)
        self.masked_instruction_encoder = nn.Embedding(vocab_size, emb_dim)
        self.state_encoder_projection = nn.Linear(7 * emb_dim, emb_dim)
        self.transformer = nn.Transformer(
            d_model=emb_dim,
            dim_feedforward=emb_dim * 4,
            nhead=nhead,
            num_encoder_layers=nlayers,
            num_decoder_layers=nlayers,
            norm_first=norm_first,
            dropout=dropout_p,
        )
        self.out_projection = nn.Linear(emb_dim, vocab_size)
        self.vocab_size = vocab_size
        self.pad_word_idx = pad_word_idx

        self.timesteps = timesteps

        (
            betas,
            alphas,
            sqrt_recip_alphas,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
            posterior_variance,
        ) = compute_schedule_variables(self.timesteps)
        self.betas = betas
        self.alphas = alphas
        self.sqrt_recip_alphas = sqrt_recip_alphas
        self.sqrt_alphas_cumprod = sqrt_alphas_cumprod
        self.sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod
        self.posterior_variance = posterior_variance

    def configure_optimizers(self):
        return transformer_optimizer_config(
            self,
            self.hparams.lr,
            weight_decay=self.hparams.wd,
            decay_power=self.hparams.decay_power,
            warmup_proportion=self.hparams.warmup_proportion,
        )

    def forward(self, x, override_p_image_drop=None):
        masked_instruction, noisy_instruction, state, timestep_idx = x

        null_state = torch.zeros_like(state)
        drop_state_indicator = (
            torch.arange(state.shape[0], device=self.device)
            < (
                (
                    override_p_image_drop
                    if override_p_image_drop is not None
                    else self.hparams.p_image_drop
                )
                * state.shape[0]
            )
        )[:, None, None]
        drop_state = null_state * drop_state_indicator + state * (~drop_state_indicator)

        # Masked instruction is part of the input,
        # we decode the encoded_instruction
        encoded_masked_instruction = self.masked_instruction_encoder(masked_instruction)
        encoded_masked_instruction = encoded_masked_instruction + self.pe_instruction(
            encoded_masked_instruction
        )

        encoded_state = self.state_encoder(drop_state)
        projected_state = self.state_encoder_projection(encoded_state)
        encoded_instruction = self.projection_instructions(noisy_instruction)
        encoded_instruction = encoded_instruction + self.pe_instruction(
            encoded_instruction
        )
        timestep_embedding = self.time_embedding(timestep_idx)

        decoded_instruction = self.transformer(
            torch.cat(
                [
                    encoded_masked_instruction,
                    projected_state,
                ],
                dim=1,
            ).transpose(0, 1),
            torch.cat(
                [encoded_instruction, timestep_embedding[:, None]], dim=-2
            ).transpose(0, 1),
            src_key_padding_mask=torch.cat(
                [
                    masked_instruction == self.pad_word_idx,
                    torch.zeros_like(state[..., 0]).bool(),
                ],
                dim=-1,
            ),
        )[:-1].transpose(0, 1)

        return self.out_projection(decoded_instruction)

    def training_step(self, x, idx):
        instruction, state = x
        encoded_instruction = F.one_hot(instruction, self.vocab_size).float()

        # Lets mask x% of the tokens in the instruction, BERT style
        masked_instruction = make_masked_instruction(
            instruction,
            self.pad_word_idx,
            mask_pct=self.hparams.mask_pct,
            contiguous_masking=self.hparams.contiguous_masking,
            device=self.device,
        )

        # Sample timesteps
        timestep_idx = torch.randint(
            0, self.timesteps, (instruction.shape[0],), device=self.device
        ).long()

        # Make noisy instrunctions
        instruction_noise = torch.randn_like(encoded_instruction)
        noisy_instructions = add_q_noise(
            x_start=encoded_instruction,
            t=timestep_idx,
            sqrt_alphas_cumprod=self.sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod,
            noise=instruction_noise,
        )

        predicted_noise = self.forward(
            (masked_instruction, noisy_instructions, state, timestep_idx)
        )
        loss = F.smooth_l1_loss(instruction_noise, predicted_noise)
        self.log("loss", loss)

        return loss

    def predict_step(self, x, idx):
        instruction, state = x

        expand_instruction = (
            instruction[:, None]
            .expand(-1, self.hparams.instruction_samples, instruction.shape[-1])
            .flatten(0, 1)
        )
        expand_instruction_one_hot = F.one_hot(
            expand_instruction, self.vocab_size
        ).float()
        state = (
            state[:, None]
            .expand(
                -1, self.hparams.instruction_samples, state.shape[-2], state.shape[-1]
            )
            .flatten(0, 1)
        )

        instruction_noise = torch.randn_like(expand_instruction_one_hot)

        stacked_preds = torch.stack(
            p_sample_loop(
                self,
                state,
                expand_instruction,
                instruction_noise,
                self.timesteps,
                self.betas,
                self.sqrt_one_minus_alphas_cumprod,
                self.sqrt_recip_alphas,
                self.posterior_variance,
                pad_word_idx=self.pad_word_idx,
                contiguous_masking=self.hparams.contiguous_masking,
                mask_pct=self.hparams.mask_pct,
            )
        )
        return stacked_preds.view(
            stacked_preds.shape[0],
            instruction.shape[0],
            self.hparams.instruction_samples,
            instruction.shape[-1],
            -1,
        ).permute(1, 2, 0, 3, 4)


class MapDataset(Dataset):
    def __init__(self, dataset, func):
        super().__init__()
        self.dataset = dataset
        self.func = func

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.func(self.dataset[idx])


def rebalance_training_data(training_data, WORD2IDX, IDX2WORD):
    training_data_indices_by_command = {}
    for i in range(len(training_data)):
        if WORD2IDX["cautiously"] in training_data[i][0]:
            continue

        cmd = " ".join(map(lambda x: IDX2WORD[x], training_data[i][0]))
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
                [training_data[i] for i in x[:min_len]]
                for x in training_data_indices_by_command.values()
            ]
        )
    )

    return balanced_training_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-demonstrations", type=str, required=True)
    parser.add_argument("--valid-demonstrations-directory", type=str, required=True)
    parser.add_argument("--dictionary", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--valid-batch-size", type=int, default=128)
    parser.add_argument("--timesteps", type=int, default=2000)
    parser.add_argument("--batch-size-mult", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--nlayers", type=int, default=8)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dropout-p", type=float, default=0.0)
    parser.add_argument("--norm-first", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--warmup-proportion", type=float, default=0.1)
    parser.add_argument("--decay-power", type=int, default=-1)
    parser.add_argument("--iterations", type=int, default=2500000)
    parser.add_argument("--check-val-every", type=int, default=1000)
    parser.add_argument("--enable-progress", action="store_true")
    parser.add_argument("--restore-from-checkpoint", action="store_true")
    parser.add_argument("--version", type=int, default=None)
    parser.add_argument("--tag", type=str, default="none")
    args = parser.parse_args()

    exp_name = "instruction_diffsion_gscan"
    model_name = (
        f"transformer_encoder_only_l_{args.nlayers}_h_{args.nhead}_d_{args.hidden_size}"
    )
    dataset_name = "gscan"
    effective_batch_size = args.train_batch_size * args.batch_size_mult
    exp_name = f"{exp_name}_s_{args.seed}_m_{model_name}_it_{args.iterations}_b_{effective_batch_size}_d_gscan_t_{args.tag}_drop_{args.dropout_p}"
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

    pl.seed_everything(seed)

    (
        (
            WORD2IDX,
            ACTION2IDX,
            color_dictionary,
            noun_dictionary,
        ),
        (train_demonstrations, valid_demonstrations_dict),
    ) = load_data(
        args.train_demonstrations, args.valid_demonstrations_directory, args.dictionary
    )

    IDX2WORD = {i: w for w, i in WORD2IDX.items()}
    IDX2ACTION = {i: w for w, i in ACTION2IDX.items()}

    pad_word = WORD2IDX["[pad]"]
    pad_action = ACTION2IDX["[pad]"]
    sos_action = ACTION2IDX["[sos]"]
    eos_action = ACTION2IDX["[eos]"]

    balanced_training_data = rebalance_training_data(
        train_demonstrations, WORD2IDX, IDX2WORD
    )

    train_dataset = ReshuffleOnIndexZeroDataset(
        PaddingDataset(
            MapDataset(balanced_training_data, lambda x: (x[0], x[2])),
            (8, None),
            (WORD2IDX["[pad]"], None),
        )
    )

    pl.seed_everything(seed)
    meta_module = InstructionDiffusionModel(
        len(IDX2WORD),
        args.timesteps,
        lr=args.lr,
        emb_dim=args.hidden_size,
        dropout_p=args.dropout_p,
        nlayers=args.nlayers,
        nhead=args.nhead,
        norm_first=args.norm_first,
        decay_power=args.decay_power,
        warmup_proportion=args.warmup_proportion,
    )
    print(meta_module)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        pin_memory=True,
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

    logs_root_dir = f"logs/{exp_name}/{model_name}/{dataset_name}/{seed}"
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
        precision=16 if torch.cuda.is_available() else None,
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
                    Subset(demonstrations, torch.randperm(len(demonstrations))[:1024]),
                    (8, 72, None),
                    (pad_word, pad_action, None),
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
