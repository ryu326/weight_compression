import codecs
import gzip
import lzma
from typing import IO, Union

import numpy as np
import torch


def get_int(b: bytes) -> int:
    return int(codecs.encode(b, "hex"), 16)


def open_maybe_compressed_file(path: Union[str, IO]) -> Union[IO, gzip.GzipFile]:
    """Return a file object that possibly decompresses 'path' on the fly.
    Decompression occurs when argument `path` is a string and ends with '.gz' or '.xz'.
    """
    if not isinstance(path, torch._six.string_classes):
        return path
    if path.endswith(".gz"):
        return gzip.open(path, "rb")
    if path.endswith(".xz"):
        return lzma.open(path, "rb")
    return open(path, "rb")


SN3_PASCALVINCENT_TYPEMAP = {
    8: (torch.uint8, np.uint8, np.uint8),
    9: (torch.int8, np.int8, np.int8),
    11: (torch.int16, np.dtype(">i2"), "i2"),
    12: (torch.int32, np.dtype(">i4"), "i4"),
    13: (torch.float32, np.dtype(">f4"), "f4"),
    14: (torch.float64, np.dtype(">f8"), "f8"),
}


def read_sn3_pascalvincent_tensor(path: Union[str, IO], strict: bool = True) -> torch.Tensor:
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
    Argument may be a filename, compressed filename, or file object.
    """
    # read
    with open_maybe_compressed_file(path) as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert 1 <= nd <= 3
    assert 8 <= ty <= 14
    m = SN3_PASCALVINCENT_TYPEMAP[ty]
    s = [get_int(data[4 * (i + 1) : 4 * (i + 2)]) for i in range(nd)]
    parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
    assert parsed.shape[0] == np.prod(s) or not strict
    return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)


def read_label_file(path: str) -> torch.Tensor:
    with open(path, "rb") as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert x.dtype == torch.uint8
    assert x.ndimension() == 1
    return x.long()


def read_image_file(path: str) -> torch.Tensor:
    with open(path, "rb") as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert x.dtype == torch.uint8
    assert x.ndimension() == 3
    return x
