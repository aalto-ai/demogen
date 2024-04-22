import argparse

from gscan_metaseq2seq.util.load_data import load_data_directories
from gscan_metaseq2seq.util.dataset import (
    PaddingDataset,
    ReorderSupportsByDistanceDataset,
    MapDataset,
)
from gscan_metaseq2seq.models.enc_dec_transformer.enc_dec_transformer_model import (
    TransformerLearner,
)

from analyze_failure_cases import get_transformer_predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dictionary", type=str, required=True)
    parser.add_argument("--data-directory", type=str, required=True)
    parser.add_argument("--transformer-checkpoint", type=str, required=True)
    parser.add_argument("--disable-cuda", action="store_true")
    parser.add_argument("--limit-load", type=int, default=None)
    parser.add_argument("--pad-instructions-to", type=int, default=8)
    parser.add_argument("--pad-actions-to", type=int, default=128)
    parser.add_argument("--pad-state-to", type=int, default=36)
    parser.add_argument("--metalearn-demonstrations-limit", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    (
        (
            WORD2IDX,
            ACTION2IDX,
            color_dictionary,
            noun_dictionary,
        ),
        (train_demonstrations, valid_demonstrations_dict),
    ) = load_data_directories(
        args.data_directory, args.dictionary, limit_load=args.limit_load
    )

    IDX2WORD = {i: w for w, i in WORD2IDX.items()}
    IDX2ACTION = {i: w for w, i in ACTION2IDX.items()}

    pad_word = WORD2IDX["[pad]"]
    pad_action = ACTION2IDX["[pad]"]
    pad_state = 0
    sos_action = ACTION2IDX["[sos]"]
    eos_action = ACTION2IDX["[eos]"]

    transformer_dataset = PaddingDataset(
        valid_demonstrations_dict["h"],
        (32, 128, (36, 7)),
        (pad_word, pad_action, 0),
    )

    (
        transformer_predicted_targets_stacked,
        transformer_logits_stacked,
        transformer_exacts_stacked,
    ) = get_transformer_predictions(
        args.transformer_checkpoint, transformer_dataset, not args.disable_cuda
    )

    import pdb
    pdb.set_trace()