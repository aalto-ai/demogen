To the reviewer,

Thank you for taking the time to review our submission!

This zip file contains the code used to generate the results in the main paper.

# Dependencies

 - pytorch, pytorch-lightning, numpy.
 - You can find exact requirements in `requirements.txt`

# Setup

In a virtual environment, run `python setup.py develop`.

# Generating Data

You will need to generate the data first. To do that, use the `generate_data.py` script. You
will need a copy of the [gSCAN compositional splits](https://github.com/LauraRuis/groundedSCAN/blob/master/data/compositional_splits.zip).

Invoke it like this:

    python scripts/generate_data.py
           --gscan-dataset path/to/compositional_splits/dataset.txt
           --output-directory data/baseline
           --generate-mode baseline

    python scripts/generate_data.py
           --gscan-dataset path/to/compositional_splits/dataset.txt
           --output-directory data/metalearn
           --generate-mode metalearn


There are a few different `--generate-mode` options:

 - `baseline`: No-metalearning, to be used with `train_transformer.py`
 - `metalearn`: Metalearning with oracle instructions and actions
 - `metalearn_distractors`: Metalearning with oracle instructions and actions, plus three distractor instructions
 - `metalearn_random_only`: Metalearning but with random oracle instructions and actions
 - `metalearn_sample_environments`: Metalearning but sample environments instead of generating actions for the same environment with the oracle
 - `metalearn_transformer_actions`: Metalearning but use the transformer passed in `--transformer` to generate the actions from oracle instructions

# Training the models

To train the meta-learning models, use something like:

    python scripts/train_meta_seq2seq_transformer.py \
    --train-demonstrations data/metalearn/train.pb \
    --valid-demonstrations data/metalearn/valid \
    --dictionary data/baseline/dictionary.pb \
    --seed 0
    --train-batch-size 32 \
    --valid-batch-size 32 \
    --batch-size-mult 4 \
    --iterations 100 \
    --version 100 \
    --enable-progress


To train the baseline models, use something like:

    python scripts/train_transformer.py \
    --train-demonstrations data/baseline/train.pb \
    --valid-demonstrations data/baseline/valid \
    --dictionary data/baseline/dictionary.pb \
    --seed 0
    --train-batch-size 32 \
    --valid-batch-size 32 \
    --batch-size-mult 4 \
    --iterations 100 \
    --version 100 \
    --enable-progress

You might want to use a large `--batch-size-mult` to get large effective batch sizes like in the paper.

Logs (both tensorboard and csv logs) are automatic and go to `logs/gscan_s_{seed}_m_{model_name}_it_{iterations}_b_{effective_batch_size}_d_{dataset_name}_t_{tag}/{model_name}/{dataset_name}/{seed}/lightning_logs/version_{version}`

# Analyzing the results and reproducing the Tables in the main paper.

Assuming that you run over:

 - seeds 0 through 9
 - datasets:
    - `baseline`: (with name `gscan`)
    - `metalearn`: (with name `gscan_metalearn_fixed`)
    - `metalearn` (with `--disable-shuffle`): (with name `gscan_metalearn_fixed` and `--tag` `--noshuffle`)
    - `metalearn_distractors`: (with name `gscan_metalearn_distractors_fixed`)
    - `metalearn_transformer_actions`: (with name `gscan_imagine_actions_fixed`)
    - `metalearn_random_only`: (with name `gscan_only_random`)
    - `metalearn_sample_environments`: (with name `gscan_sample_environments_fixed`)

then you can run the `analyze_results.py` script on your `logs` dir with `--logs-dir logs`. This will open all the
logs, exclude the worst seeds and generate the tables.

    python scripts/analyze_results.py --logs-dir path/to/logs --other-scripts-logs-dir path/to/logs

# Performing the failure case analysis

This can all be found in the `analyze_failure_cases.py` script. To run this you will need a
trained meta-seq2seq model and transformer model.

    python scripts/analyze_failure_cases.py
    --compositional-splits path/to/gscan/compositional_splits/dataset.txt
    --metalearn-data-directory data/metalearn
    --baseline-data-directory data/baseline
    --meta-seq2seq-checkpoint path/to/metaseq2seq.ckp
    --transformer-checkpoint path/to/transformer.ckpt

The plots, `comparison_edit_distance_mistakes.pdf`, `num_pulls_vs_edit_distance.pdf` and `pulls_vs_edit_distance_violinplot.pdf` get saved in the current directory.
