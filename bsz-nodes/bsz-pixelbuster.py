#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os.path
import ctypes
import numpy
from sys import platform

# Pretty sure MacOS is .so as well
LIBRARY = "pixelbuster.dll" if platform == "win32" else "libpixelbuster.so"

pb_lib = None
try:
    pb_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.realpath(__file__)), LIBRARY))
    pb_lib.pb_help_ffi.restype = ctypes.c_char_p

    pb_lib.pixelbuster_ffi.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, numpy.ctypeslib.ndpointer(ndim=1, flags=('W', 'C', 'A')), ctypes.c_uint, ctypes.c_uint
    ]

    pb_lib.pixelbuster_ffi_ext.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, numpy.ctypeslib.ndpointer(ndim=1, flags=('W', 'C', 'A')), ctypes.c_uint, ctypes.c_uint,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float
    ]
except Exception as e:
    print(f"\nbsz-cui-extras: Could not load pixelbuster library '{LIBRARY}' for platform '{platform}'")
    print("bsz-cui-extras: Consider downloading the appropriate pixelbuster library for your platform from")
    print("bsz-cui-extras: https://github.com/Beinsezii/pixelbuster")
    raise e

HELP = pb_lib.pb_help_ffi().decode('UTF-8')

import nodes
import comfy_extras.nodes_clip_sdxl as nodes_xl
import comfy.samplers as samplers

DEFAULT="""\
# See the 'BSZ Pixelbuster Help'
# node for documentation
#
# External value sliders for
# this node are as follows
# e1 e2 e3 0.0 -> 1.0
# e4 e5 -1.0 -> 1.0
# e6 e7 0.0 -> 100.0
# e8 e9 -100.0 -> 100.0

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

DEFAULT_LATENT="""\
# Do not use change colorspace! Latent ≠ pixels.
# Use c1, c2, c3, c4 for channels instead of RGBA

c1 = xnorm
c1 * pi
c1 cos c1
c1 * 15
"""

class BSZPixelbuster:
    # {{{
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "e1": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "e2": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "e3": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "e4": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "e5": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "e6": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 5.0
                }),
                "e7": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 5.0
                }),
                "e8": ("FLOAT", {
                    "default": 0.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 5.0
                }),
                "e9": ("FLOAT", {
                    "default": 0.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 5.0
                }),
            },
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

    CATEGORY = "beinsezii/image"

    def pixelbuster(
        self, image, code: str,
        e1=None, e2=None, e3=None, e4=None, e5=None, e6=None, e7=None, e8=None, e9=None
    ):
        if len(code.strip()) == 0:
            return (image,)
        externals = [e if e is not None else 0.0 for e in [e1, e2, e3, e4, e5, e6, e7, e8, e9]]
        image = image.cpu().clone() # needs to clone or else the comfyui cache gets polluted
        batch_size, height, width, channels = image.shape
        for i in image:
            ndarr = i.numpy()
            buff = numpy.pad(ndarr, pad_width=((0, 0), (0, 0), (0, 1)), constant_values=1).flatten()
            pb_lib.pixelbuster_ffi_ext(
                code.encode('UTF-8'),
                "lrgba".encode('UTF-8'),
                buff,
                buff.nbytes,
                width,
                *externals
            )
            ndarr[:] = buff.reshape(height, width, channels+1)[:, :, :-1]
            del buff
        return (image,)
    # }}}

class BSZLatentbuster:
    # {{{
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "e1": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "e2": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "e3": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "e4": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "e5": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "e6": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 5.0
                }),
                "e7": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 5.0
                }),
                "e8": ("FLOAT", {
                    "default": 0.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 5.0
                }),
                "e9": ("FLOAT", {
                    "default": 0.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 5.0
                }),
            },
            "required": {
                "latent": ("LATENT",),
                "code": ("STRING", {
                    "multiline": True,
                    "default": DEFAULT_LATENT,
                }),
            },
        }

    RETURN_TYPES = ("LATENT",)
    # RETURN_NAMES = ("image",)

    FUNCTION = "latentbuster"

    #OUTPUT_NODE = False

    CATEGORY = "beinsezii/latent/advanced"

    def latentbuster(
        self, latent, code: str,
        e1=None, e2=None, e3=None, e4=None, e5=None, e6=None, e7=None, e8=None, e9=None
    ):
        if len(code.strip()) == 0:
            return (latent,)
        externals = [e if e is not None else 0.0 for e in [e1, e2, e3, e4, e5, e6, e7, e8, e9]]
        latent = latent.copy()
        samples = latent['samples'].cpu().clone()
        batch_size, channels, height, width = samples.shape
        for batch in samples:
            ndarr = batch.numpy()
            buff = ndarr.swapaxes(1, 2).reshape(channels*height*width, order='F')
            pb_lib.pixelbuster_ffi_ext(
                code.encode('UTF-8'),
                "lrgba".encode('UTF-8'),
                buff,
                buff.nbytes,
                width,
                *externals
            )
            ndarr[:] = buff.reshape(channels, height, width, order='F').swapaxes(1, 2)
        latent['samples'] = samples
        return (latent,)
    # }}}

class BSZPixelbusterHelp:
    RETURN_TYPES = ()
    CATEGORY = "beinsezii/image"
    # FUNCTION = "pixelbuster"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "help": ("STRING", {
                    "multiline": True,
                    "default": HELP
                }),
            },
        }

NODE_CLASS_MAPPINGS = {
    "BSZPixelbuster": BSZPixelbuster,
    "BSZLatentbuster": BSZLatentbuster,
    "BSZPixelbusterHelp": BSZPixelbusterHelp
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BSZPixelbuster": "BSZ Pixelbuster",
    "BSZLatentbuster": "BSZ Latentbuster",
    "BSZPixelbusterHelp": "BSZ Pixelbuster Help"
}
