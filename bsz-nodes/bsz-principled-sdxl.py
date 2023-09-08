import nodes, folder_paths
import comfy_extras.nodes_clip_sdxl as nodes_xl
import comfy_extras.nodes_upscale_model as nodes_scale
import comfy.samplers as samplers

DEBUG=False
METHODS_LATENT = { f"latent {x}": x for x in nodes.LatentUpscale.upscale_methods }
METHODS_PIXEL = { f"pixel {x}": x for x in nodes.ImageScale.upscale_methods }
METHODS_MODEL = { f"model {x}": x for x in folder_paths.get_filename_list("upscale_models")}

def roundint(n: int, step: int) -> int:
    if n % step >= step/2:
        return int(n + step - (n % step))
    else:
        return int(n - (n % step))

class BSZPrincipledSDXL:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "refiner_model": ("MODEL",),
                "refiner_clip": ("CLIP",),
                "pixel_scale_vae": ("VAE",),
            },
            "required": {
                "base_model": ("MODEL",),
                "base_clip": ("CLIP",),
                "latent_image": ("LATENT",),
                "positive_prompt_G": ("STRING", {
                    "multiline": True,
                    "default": "analogue photograph of a kitten"
                }),
                "positive_prompt_L": ("STRING", {
                    "multiline": True,
                    "default": "kitten, photograph, analogue film"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "cropped, blurry"
                }),
                "steps": ("INT", {"default": 30, "min": 0, "max": 10000}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "refiner_amount": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "refiner_ascore_positive": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "refiner_ascore_negative": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "sampler": (samplers.KSampler.SAMPLERS, {"default": "ddim"}),
                "scheduler": (samplers.KSampler.SCHEDULERS,),
                "scale_method": (["disable"] + list(METHODS_LATENT.keys()) + list(METHODS_PIXEL.keys()) + list(METHODS_MODEL.keys()),),
                "scale_target_width": ("INT", {"default": 1024, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "scale_target_height": ("INT", {"default": 1024, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "scale_denoise": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "scale_steps": ("INT", {"default": 30, "min": 0, "max": 10000}),
                "vae_tile": (["disable", "encode", "decode", "enable"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    # RETURN_NAMES = ("latent",)

    FUNCTION = "principled_sdxl"

    #OUTPUT_NODE = False

    CATEGORY = "advanced"

    def principled_sdxl(
        self,
        base_model,
        base_clip,
        latent_image,
        positive_prompt_G: str,
        positive_prompt_L: str,
        negative_prompt: str,
        steps: int,
        denoise: float,
        cfg: float,
        refiner_amount: float,
        refiner_ascore_positive: float,
        refiner_ascore_negative: float,
        sampler,
        scheduler,
        scale_method,
        scale_target_width: int,
        scale_target_height: int,
        scale_denoise: float,
        scale_steps: int,
        vae_tile: str,
        seed: int,
        refiner_model=None,
        refiner_clip=None,
        pixel_scale_vae=None,
    ):
    #{{{
        height, width = latent_image["samples"].size()[2:4]
        height *= 8
        width *= 8

        # target base resolution for least jank
        ratio = width/height
        target_width = roundint((1024 ** 2 * ratio) ** 0.5, 8)
        target_height = roundint((1024 ** 2 / ratio) ** 0.5, 8)


        # disable refiner if not provided
        if refiner_model is None and refiner_clip is None:
            refiner_amount = 0
        elif refiner_model is not None and refiner_clip is not None:
            pass
        else:
            raise Exception("You must provide both refiner model and refiner clip to use the refiner")

        # put conditioning in lambdas so they lazy-load
        # {{{
        base_pos_cond = lambda: nodes_xl.CLIPTextEncodeSDXL.encode(
            None,
            base_clip,
            width,
            height,
            0,
            0,
            target_width,
            target_height,
            positive_prompt_G,
            positive_prompt_L,
        )[0]
        base_neg_cond = lambda: nodes_xl.CLIPTextEncodeSDXL.encode(
            None,
            base_clip,
            width,
            height,
            0,
            0,
            target_width,
            target_height,
            negative_prompt,
            negative_prompt,
        )[0]
        refiner_pos_cond = lambda: nodes_xl.CLIPTextEncodeSDXLRefiner.encode(
            None,
            refiner_clip,
            refiner_ascore_positive,
            width, # should these be target?
            height,
            positive_prompt_G
        )[0]
        refiner_neg_cond = lambda: nodes_xl.CLIPTextEncodeSDXLRefiner.encode(
            None,
            refiner_clip,
            refiner_ascore_negative,
            width,
            height,
            negative_prompt
        )[0]
        # }}}

        # internal fn for scope access. Returns latent
        def maybe_refine(latent_image):
        # {{{
            if DEBUG: print(f"""
    Running maybe_refine() with vars
        width: {width}
        height: {height}
        scale_target_width: {scale_target_width}
        scale_target_height: {scale_target_height}
        denoise: {denoise}
        """)
            # steps skipped by img2img are effectively base steps as far
            # as the refiner is concerned
            adjusted_refiner_amount = min(1, refiner_amount / max(denoise, 0.00001))
            base_start = round(steps - steps * denoise)
            base_end = round((steps - base_start) * ( 1 - adjusted_refiner_amount) + base_start)

            base_run = False

            if base_start < base_end:
                if DEBUG: print(f"Running Base - total: {steps} start: {base_start} end: {base_end}")
                latent_image = nodes.common_ksampler(
                    base_model,
                    seed,
                    steps,
                    cfg,
                    sampler,
                    scheduler,
                    base_pos_cond(),
                    base_neg_cond(),
                    latent_image,
                    start_step=base_start,
                    last_step=None if base_end == steps else base_end,
                    force_full_denoise=False if base_end < steps else True,
                )[0]
                base_run = True

            if base_end < steps:
                if DEBUG: print(f"Running Refiner - total: {steps} start: {base_end}")
                latent_image = nodes.common_ksampler(
                    refiner_model,
                    seed,
                    steps,
                    cfg,
                    sampler,
                    scheduler,
                    refiner_pos_cond(),
                    refiner_neg_cond(),
                    latent_image,
                    start_step=base_end,
                    force_full_denoise=True,
                    disable_noise=base_run
                )[0]

            return latent_image
        # }}}

        # High Res Fix. Can technically do low res too I guess, so it's "scale" not "upscale"
        if scale_method != "disable" and (width != scale_target_width or height != scale_target_height):
            if DEBUG: print(f"Running initial low-res pass")
            if scale_method not in METHODS_LATENT and pixel_scale_vae is None:
                raise Exception(f'To use scale method "{scale_method}" you must provide a VAE')

            scale_model = None
            if scale_method in METHODS_MODEL:
                scale_model = nodes_scale.UpscaleModelLoader.load_model(None, METHODS_MODEL[scale_method])[0]

            latent_image = maybe_refine(latent_image)

            if DEBUG: print(f"Upscaling with method {scale_method}")
            if scale_method in METHODS_LATENT:
                latent_image = nodes.LatentUpscale.upscale(None, latent_image, METHODS_LATENT[scale_method], scale_target_width, scale_target_height, "disabled")[0]
            else:
                decoder = nodes.VAEDecodeTiled() if vae_tile == "enable" or vae_tile == "decode" else nodes.VAEDecode()
                pixels = decoder.decode(pixel_scale_vae, latent_image)[0]
                del decoder
                if scale_method in METHODS_PIXEL:
                    pixels = nodes.ImageScale.upscale(None, pixels, METHODS_PIXEL[scale_method], scale_target_width, scale_target_height, "disabled")[0]
                elif scale_method in METHODS_MODEL:
                    pixels = nodes_scale.ImageUpscaleWithModel.upscale(None, scale_model, pixels)[0]
                    pixels = nodes.ImageScale.upscale(None, pixels, 'bicubic', scale_target_width, scale_target_height, "disabled")[0]
                else:
                    raise ValueError("Unreachable!")

                encoder = nodes.VAEEncodeTiled() if vae_tile == "enable" or vae_tile == "encode" else nodes.VAEEncode()
                latent_image = encoder.encode(pixel_scale_vae, pixels)[0]
                del pixels, encoder

            denoise = scale_denoise
            steps = scale_steps
            width, height = scale_target_width, scale_target_height

            # might not be necessary
            del scale_model

        if DEBUG: print(f"Running main pass")
        latent_image = maybe_refine(latent_image)
        return (latent_image,)
    #}}}

NODE_CLASS_MAPPINGS = {"BSZPrincipledSDXL": BSZPrincipledSDXL}
NODE_DISPLAY_NAME_MAPPINGS = {"BSZPrincipledSDXL": "BSZ Principled SDXL"}
