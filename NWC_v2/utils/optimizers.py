"""Main + aux optimizer split.  Mirrors NWC.utils.optimizers.configure_optimizers.

Aux params = anything whose name contains `.quantiles` (compressai/lattice EB
quantiles).  Everything else goes to the main Adam.
"""
import torch.optim as optim


def configure_optimizers(net, args):
    parameters = {
        n for n, p in net.named_parameters() if ".quantiles" not in n and p.requires_grad
    }
    aux_parameters = {
        n for n, p in net.named_parameters() if ".quantiles" in n and p.requires_grad
    }

    params_dict = dict(net.named_parameters())

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=float(args.learning_rate),
    )
    if aux_parameters:
        aux_optimizer = optim.Adam(
            (params_dict[n] for n in sorted(aux_parameters)),
            lr=float(args.aux_learning_rate),
        )
    else:
        aux_optimizer = None
    return optimizer, aux_optimizer
