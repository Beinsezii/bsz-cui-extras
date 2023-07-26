import nodes
import comfy_extras.nodes_clip_sdxl as nodes_xl
import comfy.samplers as samplers

class BSZPrincipledSDXL:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "latent_image": ("LATENT",),
            },
            "required": {
                "base_model": ("MODEL",),
                "base_clip": ("CLIP",),
                "refiner_model": ("MODEL",),
                "refiner_clip": ("CLIP",),
                "positive_prompt_G": ("STRING", {
                    "multiline": False, # True means latent preview is tiny
                    "default": "photograph of a kitten"
                }),
                "positive_prompt_L": ("STRING", {
                    "multiline": False,
                    "default": "analogue film"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": False,
                    "default": "cropped, blurry"
                }),
                "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 100.0}),
                "refiner_amount": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "refiner_ascore_positive": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "refiner_ascore_negative": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "refiner_misalign_steps": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "width": ("INT", {"default": 1024, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "target_width": ("INT", {"default": 1024, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "target_height": ("INT", {"default": 1024, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "sampler": (samplers.KSampler.SAMPLERS,),
                "scheduler": (samplers.KSampler.SCHEDULERS,),
                "scale_to_target": (["disable", "enable"],),
                "scale_method": (nodes.LatentUpscale.upscale_methods, {"default": "bislerp"}),
                "scale_denoise": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
                "scale_initial_steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                "scale_initial_cutoff": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
                "scale_initial_sampler": (samplers.KSampler.SAMPLERS, {"default": "dpmpp_2m"}),
                "scale_initial_scheduler": (samplers.KSampler.SCHEDULERS, {"default": "karras"}),
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
        refiner_misalign_steps: int,
        width: int,
        height: int,
        target_width: int,
        target_height: int,
        sampler,
        scheduler,
        seed: int,
        scale_to_target: str,
        scale_method,
        scale_denoise: float,
        scale_initial_steps: int,
        scale_initial_cutoff: float,
        scale_initial_sampler,
        scale_initial_scheduler,
        latent_image=None,
    ):
        # Make latent if none provided
        if latent_image is None:
            latent_image = nodes.EmptyLatentImage.generate(self, width, height)[0]

        # Base clips
        base_pos_cond = nodes_xl.CLIPTextEncodeSDXL.encode(
            self,
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
        base_neg_cond = nodes_xl.CLIPTextEncodeSDXL.encode(
            self,
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

        # High Res Fix. Can technically do low res too I guess, so it's "scale" not "upscale"
        if scale_to_target == "enable" and (width != target_width or height != target_height):
            latent_image = nodes.common_ksampler(
                base_model,
                seed,
                scale_initial_steps,
                cfg,
                scale_initial_sampler,
                scale_initial_scheduler,
                base_pos_cond,
                base_neg_cond,
                latent_image,
                start_step=round(scale_initial_steps - denoise * scale_initial_steps),
                last_step=round(scale_initial_cutoff * scale_initial_steps),
                force_full_denoise=True,
            )[0]
            latent_image = nodes.LatentUpscale.upscale(self, latent_image, scale_method, target_width, target_height, "disabled")[0]
            # Base clips with new sizes
            base_pos_cond = nodes_xl.CLIPTextEncodeSDXL.encode(self, base_clip, target_width, target_height, 0, 0, target_width, target_height, positive_prompt_G, positive_prompt_L)[0]
            base_neg_cond = nodes_xl.CLIPTextEncodeSDXL.encode(self, base_clip, target_width, target_height, 0, 0, target_width, target_height, negative_prompt, negative_prompt)[0]

            denoise = scale_denoise

        # base/refiner split
        base_start = round(steps - steps * denoise)
        base_end = round((steps - base_start) * ( 1 - refiner_amount) + base_start)

        # Skip refiner if < 1 step
        if base_end == steps:
            return nodes.KSampler.sample(self, base_model, seed, steps, cfg, sampler, scheduler, base_pos_cond, base_neg_cond, latent_image, denoise)
        else:
            refiner_pos_cond = nodes_xl.CLIPTextEncodeSDXLRefiner.encode(self, refiner_clip, 8.0, target_width, target_height, positive_prompt_G)[0]
            refiner_neg_cond = nodes_xl.CLIPTextEncodeSDXLRefiner.encode(self, refiner_clip, 2.0, target_width, target_height, negative_prompt)[0]

            # partial base pass
            latent_image = nodes.common_ksampler(
                base_model,
                seed,
                steps,
                cfg,
                sampler,
                scheduler,
                base_pos_cond,
                base_neg_cond,
                latent_image,
                start_step=base_start,
                last_step=base_end,
                force_full_denoise=False,
            )[0]

            # refiner continue sampling after base
            return nodes.common_ksampler(
                refiner_model,
                seed,
                min(steps, steps - refiner_misalign_steps),
                cfg,
                sampler,
                scheduler,
                refiner_pos_cond,
                refiner_neg_cond,
                latent_image,
                start_step=base_end,
                disable_noise=True,
                force_full_denoise=True,
            )


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "BSZPrincipledSDXL": BSZPrincipledSDXL
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "BSZPrincipledSDXL": "BSZ Principled SDXL"
}
