import torch
import nodes

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

    CATEGORY = "latent/advanced"

    def log(self, latent):
        samples = latent['samples']
        print("\nLatent size:", list(samples.size()))
        wh = samples.size()[-1] * samples.size()[-2]
        a = samples[0][0].sum().item() / wh
        b = samples[0][1].sum().item() / wh
        c = samples[0][2].sum().item() / wh
        d = samples[0][3].sum().item() / wh
        print(f"Tensor 0 Averages:\n{[a, b, c, d]}\n")
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

    CATEGORY = "latent/advanced"

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
                    "step": 0.1,
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    # RETURN_NAMES = ("latent",)

    FUNCTION = "offset"

    #OUTPUT_NODE = False

    CATEGORY = "latent"

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
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
            "width": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
            "height": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
        }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "latent"

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
            "r": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
            "g": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
            "b": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
            "a": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
            "width": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
            "height": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
        }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "latent"

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

NODE_CLASS_MAPPINGS = {
    "BSZLatentDebug": BSZLatentDebug,
    "BSZLatentFill": BSZLatentFill,
    "BSZLatentOffsetXL": BSZLatentOffsetXL,
    "BSZColoredLatentImageXL": BSZColoredLatentImageXL,
    "BSZLatentRGBAImage": BSZLatentRGBAImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSZLatentDebug": "BSZ Latent Debug",
    "BSZLatentFill": "BSZ Latent Fill",
    "BSZLatentOffsetXL": "BSZ Latent Offset XL",
    "BSZColoredLatentImageXL": "BSZ Colored Latent Image XL",
    "BSZLatentRGBAImage": "BSZ Latent RGBA Image",
}
