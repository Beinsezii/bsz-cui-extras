class BSZLatentFill:
    # {{{
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent_image": ("LATENT",),
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

    def fill(self, latent_image, a: float, b: float, c: float, d: float):
        samples = latent_image['samples'].clone();
        for batch in samples:
            batch[0].fill_(a)
            batch[1].fill_(b)
            batch[2].fill_(c)
            batch[3].fill_(d)
        return ({'samples': samples},)
# }}}


NODE_CLASS_MAPPINGS = {
    "BSZLatentFill": BSZLatentFill,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSZLatentFill": "BSZ Latent Fill",
}
