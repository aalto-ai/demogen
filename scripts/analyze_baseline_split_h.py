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
        args.data_directory, args.dictionary, limit_load=args.limit_load, only_splits=["h"]
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

    correct_prediction_targets = [
        d for d, e in zip(transformer_dataset, transformer_exacts_stacked)
        if e
    ]

    prediction_target_has_pull = [
        ACTION2IDX['pull'] in c[1] for c in correct_prediction_targets
    ]

    prediction_target_has_spin = [
         str(ACTION2IDX['turn left']) * 4 in "".join(map(str, c[1].tolist()))
         for c in correct_prediction_targets
    ]

    target_has_pull = [
         ACTION2IDX['pull'] in c[1] for c in transformer_dataset
    ]

    target_has_spin = [
         str(ACTION2IDX['turn left']) * 4 in "".join(map(str, c[1].tolist()))
         for c in transformer_dataset
    ]

    print("Has pull from correct prediction targets: {:.4f}".format(sum(prediction_target_has_pull) / len(correct_prediction_targets)))
    print("Has pull in target from all targets: {:.4f}".format(sum(target_has_pull) / len(transformer_dataset)))

    print("Has spin from correct prediction targets: {:.4f}".format(sum(prediction_target_has_spin) / len(correct_prediction_targets)))
    print("Has spin in target from all targets: {:.4f}".format(sum(target_has_spin) / len(transformer_dataset)))


if __name__ == "__main__":
    main()