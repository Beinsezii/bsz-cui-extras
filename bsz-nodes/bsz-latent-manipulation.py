import torch
import nodes
from math import pi
from colorsys import hsv_to_rgb

XL_CONSTS = {
    "black" : [-21.675981521606445, 3.864609956741333, 2.4103028774261475, 2.579195261001587],
    "white" : [18.043685913085938, 1.7262177467346191, 9.310612678527832, -8.135881423950195],
    "red" : [-19.665550231933594, -19.79644012451172, 10.68371868133545, -12.427474021911621],
    "green" : [-3.530947685241699, 14.075841903686523, 26.489261627197266, 8.67661190032959],
    "blue" : [0.45569008588790894, 16.3455867767334, -17.67197036743164, 4.145791053771973],
    "cyan" : [12.434264183044434, 26.013031005859375, 4.298962593078613, 7.954266548156738],
    "magenta" : [-0.9616246223449707, -5.109368801116943, -12.062283515930176, -9.02152156829834],
    "yellow" : [-6.609264373779297, -10.563915252685547, 32.47910690307617, -8.209832191467285],
}

class BSZLatentDebug:
    # {{{
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ()
    # RETURN_NAMES = ("latent",)

    FUNCTION = "log"

    OUTPUT_NODE = True

    CATEGORY = "beinsezii/latent/advanced"

    def log(self, latent):
        print("\nLatent structure:", latent)
        samples = latent['samples']
        print("Sample size:", list(samples.size()))
        wh = samples.size()[-1] * samples.size()[-2]
        for n, c in enumerate(samples[0]):
            print(f"Tensor 0 Channel {n} Min: {c.min().item()} Max: {c.max().item()} Avg: {c.sum().item() / wh}")
        print()
        return ()
# }}}

class BSZLatentFill:
    # {{{
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "a": ("FLOAT", {
                    "default": 0.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.5,
                }),
                "b": ("FLOAT", {
                    "default": 0.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.5,
                }),
                "c": ("FLOAT", {
                    "default": 0.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.5,
                }),
                "d": ("FLOAT", {
                    "default": 0.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.5,
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    # RETURN_NAMES = ("latent",)

    FUNCTION = "fill"

    #OUTPUT_NODE = False

    CATEGORY = "beinsezii/latent/advanced"

    def fill(self, latent, a: float, b: float, c: float, d: float):
        samples = latent['samples'].clone();
        for batch in samples:
            batch[0].fill_(a)
            batch[1].fill_(b)
            batch[2].fill_(c)
            batch[3].fill_(d)
        return (latent | {'samples': samples},)
# }}}

class BSZLatentOffsetXL:
    # {{{
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "offset": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.05,
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    # RETURN_NAMES = ("latent",)

    FUNCTION = "offset"

    #OUTPUT_NODE = False

    CATEGORY = "beinsezii/latent"

    def offset(self, latent, offset: float):
        samples = latent['samples'].clone();
        if offset > 0:
            cols = XL_CONSTS['white']
        elif offset < 0:
            cols = XL_CONSTS['black']
            offset = abs(offset)
        for batch in samples:
            if offset != 0:
                batch.mul_(1 - offset)
                batch[0].add_(cols[0] * offset)
                batch[1].add_(cols[1] * offset)
                batch[2].add_(cols[2] * offset)
                batch[3].add_(cols[3] * offset)
        return (latent | {'samples': samples},)
# }}}

class BSZColoredLatentImageXL:
    # {{{
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "color": (list(XL_CONSTS.keys()),),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            "width": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
            "height": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
        }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "beinsezii/latent"

    def generate(self, color: str, strength: float, width: int, height: int, batch_size: int):
        samples = torch.empty([batch_size, 4, height // 8, width // 8])
        cols = XL_CONSTS[color]
        for batch in samples:
            batch[0].fill_(cols[0] * strength)
            batch[1].fill_(cols[1] * strength)
            batch[2].fill_(cols[2] * strength)
            batch[3].fill_(cols[3] * strength)
        return ({"samples":samples},)
    # }}}

class BSZLatentRGBAImage:
    # {{{
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "vae": ("VAE", ),
            "r": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "g": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "b": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "a": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            "width": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
            "height": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
        }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "beinsezii/latent"

    def generate(self, vae, r: float, g: float, b: float, a: float, width: int, height: int, batch_size: int):
        if a < 0.01:
            return ({'samples': torch.zeros([batch_size, 4, height // 8, width // 8])},)
        pixels = torch.empty([1, height, width, 3])
        view = pixels.permute(0, 3, 1, 2)
        view[0][0].fill_(r)
        view[0][1].fill_(g)
        view[0][2].fill_(b)
        del view
        encoder = nodes.VAEEncode()
        latent = encoder.encode(vae, pixels)[0]
        latent['samples'] = latent['samples'].mul(a).expand([batch_size, 4, height // 8, width // 8])
        return (latent,)
    # }}}

class BSZLatentHSVAImage:
    # {{{
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "vae": ("VAE", ),
            "h": ("INT", {"default": 0, "min": 0, "max": 360, "step": 5}),
            "s": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "v": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "a": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            "width": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
            "height": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
        }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "beinsezii/latent"

    def generate(self, vae, h: float, s: float, v: float, a: float, width: int, height: int, batch_size: int):
        r, g, b = hsv_to_rgb(h / 360, s, v)
        return BSZLatentRGBAImage.generate(self, vae, r, g, b, a, width, height, batch_size)
    # }}}

class BSZLatentGradient:
    # {{{
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "a": ("LATENT",),
                "b": ("LATENT",),
                "pattern": ([
                    "sine",
                    "sine2",
                    "circle",
                    "squircle",
                    "rings",
                ],),
                "xfrequency": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05,
                }),
                "yfrequency": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05,
                }),
                "xoffset": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.05,
                }),
                "yoffset": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.05,
                }),
                "invert": ("BOOLEAN", {"default": False},),
            }
        }

    RETURN_TYPES = ("LATENT",)
    # RETURN_NAMES = ("latent",)

    FUNCTION = "gradient"

    #OUTPUT_NODE = False

    CATEGORY = "beinsezii/latent"

    def gradient(
        self,
        a,
        b,
        pattern: str,
        xfrequency: float,
        yfrequency: float,
        xoffset: float,
        yoffset: float,
        invert: bool
    ):
        if a['samples'].shape != b['samples'].shape:
            raise ValueError(f"Latents must have equivalent shapes!\nA: {a['samples'].shape}\nB: {b['samples'].shape}")
        aamples = a['samples'].clone();
        bamples = b['samples'].clone();
        smallest = min(len(aamples), len(bamples))
        batch, channels, height, width = aamples.shape

        xnorm = lambda: torch.tensor([n / (width-1) for n in reversed(range(width))]).expand([channels, height, width]).clone()
        ynorm = lambda: torch.tensor([n / (height-1) for n in reversed(range(height))]).expand([channels, width, height]).swapaxes(1, 2).clone()

        if pattern == "sine" or pattern == "sine2":
            factor = ynorm().add(yoffset).mul(yfrequency) if pattern == "sine" else ynorm().mul(-1).add(1+yoffset).mul(yfrequency)
            factor += xnorm().add(xoffset).mul(xfrequency)
            factor.div_(2 - abs(xfrequency / 10 - yfrequency / 10))
            factor.mul_(pi)
            factor.cos_()
            factor.add_(1)
            factor.div_(2)
        elif pattern == "circle":
            factor = xnorm().add(xoffset).mul(pi).mul(xfrequency).sin()
            factor += ynorm().add(yoffset).mul(pi).mul(yfrequency).sin()
            factor.div_(2)
            factor.abs_()
        elif pattern == "squircle":
            factor = xnorm().add(xoffset).mul(pi).mul(xfrequency).sin()
            factor *= ynorm().add(yoffset).mul(pi).mul(yfrequency).sin()
            factor.abs_()
        elif pattern == "rings":
            factor = xnorm().add(xoffset).mul(pi).sin().mul(xfrequency)
            factor += ynorm().add(yoffset).mul(pi).sin().mul(yfrequency)
            factor.sin_()
            factor.abs_()
        else:
            raise ValueError("Invalid gradient pattern!")

        for na, nb in zip(aamples, bamples):
            batchfac = factor.clone()
            if invert:
                na *= batchfac
                batchfac.mul_(-1)
                batchfac.add_(1)
                nb *= batchfac
            else:
                nb *= batchfac
                batchfac.mul_(-1)
                batchfac.add_(1)
                na *= batchfac
            na += nb
            del batchfac

        return (a | {'samples': aamples},)
# }}}

class BSZHueChromaXL:
    # {{{
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "hue": ("FLOAT", {
                    "default": 0.0,
                    "min": -180.0,
                    "max": 180.0,
                }),
                "chroma": ("FLOAT", {
                    "default": 0.0,
                    "min": -100.0,
                    "max": 100.0,
                }),
                "lightness": ("FLOAT", {
                    "default": 0.0,
                    "min": -100.0,
                    "max": 100.0,
                }),
            },
        }

    RETURN_TYPES = ("LATENT",)
    # RETURN_NAMES = ("image",)

    FUNCTION = "latent_huechroma"

    #OUTPUT_NODE = False

    CATEGORY = "beinsezii/latent"

    def latent_huechroma(self, latent, hue, chroma, lightness):
        if hue == 0 and chroma == 0 and lightness == 0:
            return (latent,)
        latent = latent.copy()
        samples = latent['samples'].clone().permute(1,0,2,3)
        # Lightness
        samples[0] -= -21.675973892211914
        samples[0] *= 100 / (abs(-21.675973892211914) + 18.038631439208984)

        samples[3] -= 2.5792038440704346
        samples[3] *= -100 / (abs(-8.136277198791504) + 2.5792038440704346)

        # Naive approach until I make a backward plot
        samples[0] += lightness
        samples[3] += lightness

        samples[0] /= 100 / (abs(-21.675973892211914) + 18.038631439208984)
        samples[0] += -21.675973892211914
        samples[3] /= -100 / (abs(-8.136277198791504) + 2.5792038440704346)
        samples[3] += 2.5792038440704346

        # Hue/Chroma
        # Approx values due to lack of forward plot
        samples[1] -= 4.560664176940918
        samples[1] *= -100 / (abs(-11.767170906066895) + 4.560664176940918)
        samples[2] -= 3.3889966011047363
        samples[2] *= 100 / (18.39630699157715 + 3.3889966011047363)
        chroma_arr = (samples[1] ** 2 + samples[2] ** 2).sqrt()
        hue_arr = samples[2].atan2(samples[1]).rad2deg()

        chroma_arr += chroma
        hue_arr += hue

        samples[1] = chroma_arr * hue_arr.deg2rad().cos()
        samples[2] = chroma_arr * hue_arr.deg2rad().sin()

        samples[1] /= -100 / (abs(-11.767170906066895) + 4.560664176940918)
        samples[1] += 4.560664176940918
        samples[2] /= 100 / (18.39630699157715 + 3.3889966011047363)
        samples[2] += 3.3889966011047363

        latent['samples'] = samples.permute(1,0,2,3)
        return (latent,)
    # }}}

NODE_CLASS_MAPPINGS = {
    "BSZLatentDebug": BSZLatentDebug,
    "BSZLatentFill": BSZLatentFill,
    "BSZLatentOffsetXL": BSZLatentOffsetXL,
    "BSZColoredLatentImageXL": BSZColoredLatentImageXL,
    "BSZLatentRGBAImage": BSZLatentRGBAImage,
    "BSZLatentHSVAImage": BSZLatentHSVAImage,
    "BSZLatentGradient": BSZLatentGradient,
    "BSZHueChromaXL": BSZHueChromaXL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSZLatentDebug": "BSZ Latent Debug",
    "BSZLatentFill": "BSZ Latent Fill",
    "BSZLatentOffsetXL": "BSZ Latent Offset XL",
    "BSZColoredLatentImageXL": "BSZ Colored Latent Image XL",
    "BSZLatentRGBAImage": "BSZ Latent RGBA Image",
    "BSZLatentHSVAImage": "BSZ Latent HSVA Image",
    "BSZLatentGradient": "BSZ Latent Gradient",
    "BSZHueChromaXL": "BSZ Hue Chroma XL",
}
