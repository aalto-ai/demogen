import torch
import torch.nn.functional as F
import torch.nn as nn

from gscan_metaseq2seq.models.embedding import BOWEmbedding
from gscan_metaseq2seq.util.scheduler import transformer_optimizer_config

from positional_encodings.torch_encodings import PositionalEncoding1D
import pytorch_lightning as pl

from tqdm.auto import trange

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
        self.norm = nn.LayerNorm(embedding_dim)
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
        z_embed_seq = z_embed_seq + self.pos_encoding(z_embed_seq)

        z_embed_seq = torch.cat([state_embed_seq, z_embed_seq], dim=1)
        z_embed_seq = self.dropout(self.norm(z_embed_seq))
        padding_bits = torch.cat([state_padding_bits, z_padding_bits], dim=-1)

        encoded_seq = self.transformer_encoder(
            z_embed_seq.transpose(1, 0),
            src_key_padding_mask=padding_bits,
        )

        # bs x emb_dim, z_seq_len x bs x emb_dim
        return encoded_seq, padding_bits


class SequenceEncoderTransformer(nn.Module):
    def __init__(
        self,
        input_size,
        embedding_dim,
        nlayers,
        nhead,
        dropout_p,
        norm_first,
        pad_word_idx,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.norm = nn.LayerNorm(embedding_dim)
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

    def forward(self, z_padded):
        z_padding_bits = z_padded == self.pad_word_idx

        z_embed_seq = self.embedding(z_padded)
        z_embed_seq = self.dropout(self.norm(z_embed_seq + self.pos_encoding(z_embed_seq)))

        encoded_seq = self.transformer_encoder(
            z_embed_seq.transpose(1, 0),
            src_key_padding_mask=z_padding_bits,
        )

        # bs x emb_dim, z_seq_len x bs x emb_dim
        return encoded_seq, z_padding_bits


class DecoderTransformer(nn.Module):
    def __init__(
        self,
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
        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
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
        embedding = self.norm(embedding + self.pos_encoding(embedding))
        embedding = self.dropout(embedding)

        decoded = self.decoder(
            tgt=embedding.transpose(0, 1),
            memory=encoder_outputs,
            memory_key_padding_mask=encoder_padding,
            tgt_key_padding_mask=input_padding_bits,
            tgt_mask=nn.Transformer.generate_square_subsequent_mask(input_padding_bits.shape[-1]).to(input_padding_bits.device).bool(),
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


def compute_encoder_decoder_model_loss_and_stats(
    model, inputs, targets, sos_target_idx, pad_target_idx
):
    actions_mask = targets == pad_target_idx

    decoder_in = torch.cat(
        [torch.ones_like(targets)[:, :1] * sos_target_idx, targets], dim=-1
    )

    # Now do the training
    preds = model.forward(*inputs, decoder_in)[:, :-1]

    # Ultimately we care about the cross entropy loss
    loss = F.cross_entropy(
        preds.flatten(0, -2),
        targets.flatten().long(),
        ignore_index=pad_target_idx,
    )

    argmax_preds = preds.argmax(dim=-1)
    argmax_preds[actions_mask] = pad_target_idx
    exacts = (argmax_preds == targets).all(dim=-1).to(torch.float).mean()

    return {
        "loss": loss,
        "acc": (preds.argmax(dim=-1)[~actions_mask] == targets[~actions_mask])
        .float()
        .mean(),
        "exacts": exacts,
    }


def autoregressive_model_unroll_predictions(
    model, inputs, target, sos_target_idx, eos_target_idx, pad_target_idx, quiet=False
):
    with torch.inference_mode(), torch.autocast(device_type=str(target.device).split(":")[0], dtype=torch.float16, enabled=True):
        encodings, key_padding_mask = model.encode(*inputs)

    # Recursive decoding, start with a batch of SOS tokens
    decoder_in = torch.tensor(sos_target_idx, dtype=torch.long, device=model.device)[
        None
    ].expand(target.shape[0], 1)

    logits = []

    with torch.inference_mode(), torch.autocast(device_type=str(target.device).split(":")[0], dtype=torch.float16, enabled=True):
        for i in trange(target.shape[1], desc="Gen tgts", disable=quiet):
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


def filter_out_padding(decoded, target, logits, eos_target_idx):
    decoded = decoded.cpu().numpy()
    decoded_select_mask = decoded != -1
    decoded = [d[m] for d, m in zip(decoded, decoded_select_mask)]

    target = target.cpu().numpy()
    target = [d[d != -1] for d in target]

    logits = logits.cpu().numpy()
    logits = [l[m] for l, m in zip(logits, decoded_select_mask)]

    return tuple([decoded, logits, target])


class SequenceTransformerLearner(pl.LightningModule):
    def __init__(
        self,
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
        self.encoder = SequenceEncoderTransformer(
            x_categories,
            embed_dim,
            nlayers,
            nhead,
            dropout_p,
            norm_first,
            pad_word_idx,
        )
        self.decoder = DecoderTransformer(
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

    def encode(self, queries):
        return self.encoder(queries)

    def decode_autoregressive(self, decoder_in, encoder_outputs, encoder_padding):
        return self.decoder(decoder_in, encoder_outputs, encoder_padding)

    def forward(self, queries, decoder_in):
        encoded, encoder_padding = self.encoder(queries)
        return self.decode_autoregressive(decoder_in, encoded, encoder_padding)

    def training_step(self, x, idx):
        query, targets = x
        stats = compute_encoder_decoder_model_loss_and_stats(
            self, (query, ), targets, self.sos_action_idx, self.pad_action_idx
        )
        self.log("tloss", stats["loss"], prog_bar=True)
        self.log("texact", stats["exacts"], prog_bar=True)
        self.log("tacc", stats["acc"], prog_bar=True)

        return stats["loss"]

    def validation_step(self, x, idx, dl_idx=0):
        query, targets = x
        stats = compute_encoder_decoder_model_loss_and_stats(
            self, (query, ), targets, self.sos_action_idx, self.pad_action_idx
        )
        self.log("vloss", stats["loss"], prog_bar=True)
        self.log("vexact", stats["exacts"], prog_bar=True)
        self.log("vacc", stats["acc"], prog_bar=True)

    def predict_step(self, x, idx, dl_idx=0):
        instruction, target, state = x[:3]

        decoded, logits, exacts, _ = autoregressive_model_unroll_predictions(
            self,
            (instruction, ),
            target,
            self.sos_action_idx,
            self.eos_action_idx,
            self.pad_action_idx,
        )

        return tuple([instruction, state, decoded, logits, exacts, target] + x[3:])


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
        stats = compute_encoder_decoder_model_loss_and_stats(
            self, (state, query), targets, self.sos_action_idx, self.pad_action_idx
        )
        self.log("tloss", stats["loss"], prog_bar=True)
        self.log("texact", stats["exacts"], prog_bar=True)
        self.log("tacc", stats["acc"], prog_bar=True)

        return stats["loss"]

    def validation_step(self, x, idx, dataloader_idx=0):
        query, targets, state = x
        stats = compute_encoder_decoder_model_loss_and_stats(
            self, (state, query), targets, self.sos_action_idx, self.pad_action_idx
        )
        self.log("vloss", stats["loss"], prog_bar=True)
        self.log("vexact", stats["exacts"], prog_bar=True)
        self.log("vacc", stats["acc"], prog_bar=True)

    def predict_step(self, x, idx, dataloader_idx=0):
        instruction, target, state = x[:3]

        decoded, logits, exacts, _ = autoregressive_model_unroll_predictions(
            self,
            (state, instruction),
            target,
            self.sos_action_idx,
            self.eos_action_idx,
            self.pad_action_idx,
        )

        return tuple([instruction, state, decoded, logits, exacts, target] + x[3:])
