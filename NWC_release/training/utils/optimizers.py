import torch.optim as optim


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {n for n, p in net.named_parameters() if ".quantiles" not in n and p.requires_grad}
    aux_parameters = {n for n, p in net.named_parameters() if ".quantiles" in n and p.requires_grad}

    print(aux_parameters)  # {'module.entropy_bottleneck_z.quantiles'}

    params_dict = dict(net.named_parameters())

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    if aux_parameters:
        aux_optimizer = optim.Adam(
            (params_dict[n] for n in sorted(aux_parameters)),
            lr=args.aux_learning_rate,
        )
    else :
        aux_optimizer = None
    return optimizer, aux_optimizer
