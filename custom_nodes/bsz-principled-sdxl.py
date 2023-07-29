import nodes
import comfy_extras.nodes_clip_sdxl as nodes_xl
import comfy.samplers as samplers

DEBUG=False
METHODS_LATENT = { f"latent {x}": x for x in nodes.LatentUpscale.upscale_methods }
METHODS_PIXEL = { f"pixel {x}": x for x in nodes.ImageScale.upscale_methods }

class BSZPrincipledSDXL:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "latent_image": ("LATENT",),
                "pixel_scale_vae": ("VAE",),
            },
            "required": {
                "base_model": ("MODEL",),
                "base_clip": ("CLIP",),
                "refiner_model": ("MODEL",),
                "refiner_clip": ("CLIP",),
                "positive_prompt_G": ("STRING", {
                    "multiline": True,
                    "default": "photograph of a kitten"
                }),
                "positive_prompt_L": ("STRING", {
                    "multiline": True,
                    "default": "analogue film"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "cropped, blurry"
                }),
                "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "refiner_amount": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "refiner_ascore_positive": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "refiner_ascore_negative": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "width": ("INT", {"default": 1024, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "target_width": ("INT", {"default": 1024, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "target_height": ("INT", {"default": 1024, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "sampler": (samplers.KSampler.SAMPLERS,),
                "scheduler": (samplers.KSampler.SCHEDULERS,),
                "scale_method": (["disable"] + list(METHODS_LATENT.keys()) + list(METHODS_PIXEL.keys()),),
                "scale_denoise": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
                "scale_initial_steps": ("INT", {"default": 30, "min": 0, "max": 10000}),
                "scale_initial_cutoff": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
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
        refiner_model,
        refiner_clip,
        positive_prompt_G: str,
        positive_prompt_L: str,
        negative_prompt: str,
        steps: int,
        denoise: float,
        cfg: float,
        refiner_amount: float,
        refiner_ascore_positive: float,
        refiner_ascore_negative: float,
        width: int,
        height: int,
        target_width: int,
        target_height: int,
        sampler,
        scheduler,
        scale_method,
        scale_denoise: float,
        scale_initial_steps: int,
        scale_initial_cutoff: float,
        vae_tile: str,
        seed: int,
        latent_image=None,
        pixel_scale_vae=None,
    ):
    #{{{
        # Make latent if none provided
        if latent_image is None:
            if DEBUG: print(f"Creating latent image sized {width}x{height}")
            latent_image = nodes.EmptyLatentImage.generate(None, width, height)[0]

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
        def maybe_refine(latent_image, cutoff=1.0):
        # {{{
            if DEBUG: print(f"""
    Running maybe_refine() with vars
        width: {width}
        height: {height}
        target_width: {target_width}
        target_height: {target_height}
        denoise: {denoise}
        cutoff: {cutoff}
        """)
            # steps skipped by img2img are effectively base steps as far
            # as the refiner is concerned
            adjusted_refiner_amount = min(1, refiner_amount / max(denoise, 0.00001))
            # base refiner split applied after cutoff so refiner is entirely cut
            cut_steps = round(steps * cutoff)
            base_start = round(steps - steps * denoise)
            base_end = min(round((steps - base_start) * ( 1 - adjusted_refiner_amount) + base_start), cut_steps)

            base_run = False

            if base_start < base_end:
                if DEBUG: print(f"Running Base - total: {cut_steps} start: {base_start} end: {base_end}")
                latent_image = nodes.common_ksampler(
                    base_model,
                    seed,
                    cut_steps,
                    cfg,
                    sampler,
                    scheduler,
                    base_pos_cond(),
                    base_neg_cond(),
                    latent_image,
                    start_step=base_start,
                    last_step=None if base_end == cut_steps else base_end,
                    force_full_denoise=False if base_end < cut_steps else True,
                )[0]
                base_run = True

            if base_end < cut_steps:
                if DEBUG: print(f"Running Refiner - total: {cut_steps} start: {base_end}")
                latent_image = nodes.common_ksampler(
                    refiner_model,
                    seed,
                    cut_steps,
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

        latent_h, latent_w = latent_image["samples"].size()[2:4]

        # High Res Fix. Can technically do low res too I guess, so it's "scale" not "upscale"
        if scale_method != "disable" and (width != target_width or height != target_height or latent_w * 8 != width or latent_h * 8 != height):
            if DEBUG: print(f"Running initial low-res pass")
            assert scale_method in METHODS_LATENT or pixel_scale_vae is not None

            post_steps, steps = steps, scale_initial_steps

            latent_image = maybe_refine(latent_image, cutoff=scale_initial_cutoff)

            if DEBUG: print(f"Upscaling with method {scale_method}")
            if scale_method in METHODS_LATENT:
                latent_image = nodes.LatentUpscale.upscale(None, latent_image, METHODS_LATENT[scale_method], target_width, target_height, "disabled")[0]
            else:
                decoder = nodes.VAEDecodeTiled() if vae_tile == "enable" or vae_tile == "decode" else nodes.VAEDecode()
                pixels = decoder.decode(pixel_scale_vae, latent_image)[0]
                del decoder
                pixels = nodes.ImageScale.upscale(None, pixels, METHODS_PIXEL[scale_method], target_width, target_height, "disabled")[0]
                encoder = nodes.VAEEncodeTiled() if vae_tile == "enable" or vae_tile == "encode" else nodes.VAEEncode()
                latent_image = encoder.encode(pixel_scale_vae, pixels)[0]
                del pixels, encoder

            width, height = target_width, target_height
            steps = post_steps
            denoise = scale_denoise

        if DEBUG: print(f"Running main pass")
        latent_image = maybe_refine(latent_image)
        return (latent_image,)
    #}}}

NODE_CLASS_MAPPINGS = {"BSZPrincipledSDXL": BSZPrincipledSDXL}
NODE_DISPLAY_NAME_MAPPINGS = {"BSZPrincipledSDXL": "BSZ Principled SDXL"}
