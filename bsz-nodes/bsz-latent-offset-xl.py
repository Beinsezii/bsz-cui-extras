XL_WHITE = [18.043685913085938, 1.7262177467346191, 9.310612678527832, -8.135881423950195]
XL_BLACK = [-21.675981521606445, 3.864609956741333, 2.4103028774261475, 2.579195261001587]

class BSZLatentOffsetXL:
    # {{{
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent_image": ("LATENT",),
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

    CATEGORY = "latent/advanced"

    def offset(self, latent_image, offset: float):
        samples = latent_image['samples'].clone();
        for batch in samples:
            if offset > 0:
                batch.mul_(1 - offset)
                batch[0].add_(XL_WHITE[0] * offset)
                batch[1].add_(XL_WHITE[1] * offset)
                batch[2].add_(XL_WHITE[2] * offset)
                batch[3].add_(XL_WHITE[3] * offset)
            elif offset < 0:
                batch.mul_(1 - abs(offset))
                batch[0].add_(XL_BLACK[0] * abs(offset))
                batch[1].add_(XL_BLACK[1] * abs(offset))
                batch[2].add_(XL_BLACK[2] * abs(offset))
                batch[3].add_(XL_BLACK[3] * abs(offset))
        return (latent_image | {'samples': samples},)
# }}}


NODE_CLASS_MAPPINGS = {
    "BSZLatentOffsetXL": BSZLatentOffsetXL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSZLatentOffsetXL": "BSZ Latent Offset XL",
}
