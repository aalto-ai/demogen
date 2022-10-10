import torch
import torch.nn as nn
import torch.optim as optim


def linear_with_warmup_schedule(
    optimizer, num_warmup_steps, num_training_steps, min_lr_scale, last_epoch=-1
):
    min_lr_logscale = min_lr_scale

    def lr_lambda(current_step):
        # Scale from 0 to 1
        if current_step <= num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Scale from 1 to min_lr_scale logarithmically
        #
        # So for example, if min_lr_logscale is -3, then
        # scale goes from 0 to -3 meaning that the lr multiplier
        # goes from 1, to 1e-1 at -1, to 1e-2 at -2 to 1e-3 at -3.
        scale = min(
            1,
            float(current_step - num_warmup_steps)
            / float(num_training_steps - num_warmup_steps),
        )
        logscale = scale * min_lr_logscale
        multiplier = 10**logscale

        return multiplier

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_parameter_names(model, exclude_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, exclude_layer_types)
            if not isinstance(child, tuple(exclude_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def transformer_optimizer_config(
    harness, lr, warmup_proportion=0.14, decay_power=-2, weight_decay=0
):
    decay_parameters = get_parameter_names(harness, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in harness.named_parameters() if n in decay_parameters
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in harness.named_parameters() if n not in decay_parameters
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.AdamW(harness.parameters(), lr=lr, weight_decay=weight_decay)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": linear_with_warmup_schedule(
                optimizer,
                harness.trainer.max_steps * warmup_proportion,
                harness.trainer.max_steps,
                decay_power,
            ),
            "interval": "step",
            "frequency": 1,
        },
    }
