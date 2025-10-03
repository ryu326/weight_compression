import argparse
import csv
import functools
import glob
import itertools
import operator
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager

import matplotlib as mpl
import numpy as np
import scipy.ndimage
from PIL import Image
import torch


def jp_compress(tensor_uint8, output_p, quality, verbose=False):
    img = Image.fromarray(tensor_uint8.cpu().numpy().astype(np.uint8))
    out_path = f"{output_p}.jpg"    
    
    img.save(out_path, quality=quality)
    # dim = float(img.size[0] * img.size[1])
    bits = (8 * _jpeg_content_length(out_path))
    return out_path, bits
    
def jp_decompress(jpeg_path: str):
    if not os.path.exists(jpeg_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {jpeg_path}")

    img = Image.open(jpeg_path)
    decomp_numpy = np.array(img)
    decomp_tensor = torch.from_numpy(decomp_numpy)
    return decomp_tensor

def _jpeg_content_length(p):
    """
    Determines the length of the content of the JPEG file stored at `p` in bytes, i.e., size of the file without the
    header. Note: Note sure if this works for all JPEGs...
    :param p: path to a JPEG file
    :return: length of content
    """
    with open(p, "rb") as f:
        last_byte = ""
        header_end_i = None
        for i in itertools.count():
            current_byte = f.read(1)
            if current_byte == b"":
                break
            # some files somehow contain multiple FF DA sequences, don't know what that means
            if header_end_i is None and last_byte == b"\xff" and current_byte == b"\xda":
                header_end_i = i
            last_byte = current_byte
        # at this point, i is equal to the size of the file
        return i - header_end_i - 2  # minus 2 because all JPEG files end in FF D0