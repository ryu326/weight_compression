from .codec import NWCv2Codec


def get_model(args, scale, shift):
    return NWCv2Codec(args, scale, shift)


__all__ = ["NWCv2Codec", "get_model"]
