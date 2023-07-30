def roundint(n: int, step: int) -> int:
    if n % step >= step/2:
        return n + step - (n % step)
    else:
        return n - (n % step)

class BSZAutoHires:
    # {{{
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_model_res": ("INT", {
                    "default": 1024,
                    "min": 8,
                    "max": 4096,
                    "step": 8
                }),
                "desired_width": ("INT", {
                    "default": 1536,
                    "min": 8,
                    "max": 4096,
                    "step": 8
                }),
                "desired_height": ("INT", {
                    "default": 1536,
                    "min": 8,
                    "max": 4096,
                    "step": 8
                }),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("Lo Res Width", "Lo Res Height", "Hi Res Width", "Hi Res Height")
    FUNCTION = "hiresify"
    CATEGORY = "utils"

    def hiresify(self, base_model_res: int, desired_width: int, desired_height: int) -> (int, int, int, int):
        mpx: int = base_model_res**2
        aspect_x: float = desired_width/desired_height
        aspect_y: float = desired_height/desired_width
        step: int = 8

        # lores width, lores height, hires width, hires height
        return (
            int(roundint((mpx * aspect_x)**0.5, step)),
            int(roundint((mpx * aspect_y)**0.5, step)),
            int(roundint(desired_width, step)),
            int(roundint(desired_height, step)),
        )
# }}}


class BSZAutoHiresAspect:
    # {{{
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_model_res": ("INT", {
                    "default": 1024,
                    "min": 8,
                    "max": 4096,
                    "step": 8
                }),
                "desired_aspect_x": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.1,
                    "max": 100.0,
                    "step": 1.0
                }),
                "desired_aspect_y": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.1,
                    "max": 100.0,
                    "step": 1.0
                }),
                "scale": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.5
                }),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("Lo Res Width", "Lo Res Height", "Hi Res Width", "Hi Res Height")
    FUNCTION = "hiresify"
    CATEGORY = "utils"

    def hiresify(self, base_model_res: int, desired_aspect_x: float, desired_aspect_y: float, scale: float) -> (int, int, int, int):
        mpx: int = base_model_res**2
        width = (mpx * desired_aspect_x / desired_aspect_y) ** 0.5
        height = (mpx * desired_aspect_y / desired_aspect_x) ** 0.5
        step: int = 8

        # lores width, lores height, hires width, hires height
        return (
            int(roundint(width, step)),
            int(roundint(height, step)),
            int(roundint(width * scale, step)),
            int(roundint(height * scale, step)),
        )
# }}}


class BSZAutoHiresCombined:
    # {{{
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_model_res": ("INT", {
                    "default": 1024,
                    "min": 8,
                    "max": 4096,
                    "step": 8
                }),
                "desired_width": ("INT", {
                    "default": 1536,
                    "min": 8,
                    "max": 4096,
                    "step": 8
                }),
                "desired_height": ("INT", {
                    "default": 1536,
                    "min": 8,
                    "max": 4096,
                    "step": 8
                }),
                "use_aspect_scale": (["enable", "disable"],),
                "desired_aspect_x": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.1,
                    "max": 100.0,
                    "step": 1.0
                }),
                "desired_aspect_y": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.1,
                    "max": 100.0,
                    "step": 1.0
                }),
                "scale": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.5
                }),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("Lo Res Width", "Lo Res Height", "Hi Res Width", "Hi Res Height")
    FUNCTION = "hiresify"
    CATEGORY = "utils"

    def hiresify(self, base_model_res: int, desired_width: int, desired_height: int, use_aspect_scale: str, desired_aspect_x: float, desired_aspect_y: float, scale: float) -> (int, int, int, int):
        if use_aspect_scale == "enable":
            return BSZAutoHiresAspect.hiresify(self, base_model_res, desired_aspect_x, desired_aspect_y, scale)
        else:
            return BSZAutoHires.hiresify(self, base_model_res, desired_width, desired_height)
# }}}


NODE_CLASS_MAPPINGS = {
    "BSZAutoHires": BSZAutoHires,
    "BSZAutoHiresAspect": BSZAutoHiresAspect,
    "BSZAutoHiresCombined": BSZAutoHiresCombined,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSZAutoHires": "BSZ Automatic Hi Res (Absolute)",
    "BSZAutoHiresAspect": "BSZ Automatic Hi Res (Aspect Scale)",
    "BSZAutoHiresCombined": "BSZ Automatic Hi Res (Combined)",
}
