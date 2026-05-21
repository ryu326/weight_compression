"""Dataset factory.  Each `get_dataset_*` returns a 6-tuple
`(train_dataset, val_dataset, train_std, val_std, train_mean, val_mean)`.

Both datasets yield dicts of the form `{"weight_block": (T, I) tensor}`.
The `train_mean` is used by the codec as `shift` (NWC parity); std is
typically 1.0 for llama8b (post-normalization) and `gaussian_std` for the
synthetic dataset.
"""
from .gaussian import get_dataset_gaussian
from .llama8b import get_dataset_llama8b


def get_dataset(args):
    name = str(args.dataset)
    if name == "gaussian":
        return get_dataset_gaussian(args)
    if name == "llama8b":
        return get_dataset_llama8b(args)
    raise ValueError(f"unknown dataset '{name}'; choose from {{'gaussian', 'llama8b'}}")


__all__ = ["get_dataset"]
