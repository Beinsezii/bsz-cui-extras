#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os.path

# sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')

import ctypes
import numpy
from sys import platform

LIBRARY = {"win32": "pixelbuster.dll", "linux": "libpixelbuster.so"}
import os.path

pb_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.realpath(__file__)), LIBRARY.get(platform)))

pb_lib.pb_help_ffi.restype = ctypes.c_char_p
HELP = pb_lib.pb_help_ffi().decode('UTF-8')

pb_lib.pixelbuster_ffi.argtypes = [
    ctypes.c_char_p, ctypes.c_char_p, numpy.ctypeslib.ndpointer(ndim=1, flags=('W', 'C', 'A')), ctypes.c_uint,
    ctypes.c_uint
]

import nodes
import comfy_extras.nodes_clip_sdxl as nodes_xl
import comfy.samplers as samplers

DEBUG=False

DEFAULT="""\
LCH

v1 = xnorm
v1 * pi
v1 sin v1

v2 = ynorm
v2 * pi
v2 / 2
v2 cos v2

v1 * v2

h = v1
h * 240"""

class BSZPixelbuster:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "code": ("STRING", {
                    "multiline": True,
                    "default": DEFAULT
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    # RETURN_NAMES = ("image",)

    FUNCTION = "pixelbuster"

    #OUTPUT_NODE = False

    CATEGORY = "image/postprocessing"

    def pixelbuster(self, image, code: str):
        image = image.cpu().clone() # needs to clone or else the comfyui cache gets polluted
        batch_size, height, width, channels = image.shape
        for i in image:
            ndarr = i.numpy()
            buff = numpy.pad(ndarr, pad_width=((0, 0), (0, 0), (0, 1)), constant_values=1).flatten()
            pb_lib.pixelbuster_ffi(
                code.encode('UTF-8'),
                "lrgba".encode('UTF-8'),
                buff,
                buff.nbytes,
                width,
            )
            ndarr[:] = buff.reshape(height, width, channels+1)[:, :, :-1]
            del buff
        return (image,)

NODE_CLASS_MAPPINGS = {"BSZPixelbuster": BSZPixelbuster}
NODE_DISPLAY_NAME_MAPPINGS = {"BSZPixelbuster": "BSZ Pixelbuster"}
