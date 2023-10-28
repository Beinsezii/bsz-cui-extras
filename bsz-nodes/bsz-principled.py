import torch
import comfy
import nodes
import comfy_extras.nodes_clip_sdxl as nodes_xl
# Scale
import folder_paths
import comfy_extras.nodes_upscale_model as nodes_scale

from os import getenv

DEBUG = getenv('BSZ_CUI_DEBUG', False)

# maintain pointer to the old FN so it's never lost in errs
OLD_PREPARE_NOISE = comfy.sample.prepare_noise

METHODS_LATENT = { f"latent {x}": x for x in nodes.LatentUpscale.upscale_methods }
METHODS_PIXEL = { f"pixel {x}": x for x in nodes.ImageScale.upscale_methods }
METHODS_MODEL = { f"model {x}": x for x in folder_paths.get_filename_list("upscale_models")}

def _prepare_noise(latent_image, seed, noise_inds=None):
    b, c, h, w = latent_image.shape
    slices = []
    if noise_inds is not None:
        for n in noise_inds:
            slices.append(torch.randn([1, c, h, w], dtype=latent_image.dtype, layout=latent_image.layout, generator=torch.manual_seed(seed+n), device="cpu"))
    else:
        for n in range(seed, seed+b):
            slices.append(torch.randn([1, c, h, w], dtype=latent_image.dtype, layout=latent_image.layout, generator=torch.manual_seed(n), device="cpu"))
    return torch.cat(slices, axis=0)

def roundint(n: int, step: int) -> int:
    if n % step >= step/2:
        return int(n + step - (n % step))
    else:
        return int(n - (n % step))

class CondStage:
    def __init__(self, text: str, xl_target="1k", refiner_asc=6.0):
        self.text = text
        self.xl_target = xl_target
        self.refiner_asc = refiner_asc

    def process(self, latent, clip, other):
        return self._encode(latent, clip)

    def _encode(self, latent, clip):
        # {{{
        height, width = latent["samples"].size()[2:4]
        height *= 8
        width *= 8

        # target base resolution for least jank
        ratio = width/height
        target_width, target_height = width, height
        if self.xl_target == "1k":
            target_width = roundint((1024 ** 2 * ratio) ** 0.5, 8)
            target_height = roundint((1024 ** 2 / ratio) ** 0.5, 8)
        elif self.xl_target == "4k":
            if width > height:
                target_width = 4096
                target_height = roundint(4096 / ratio, 8)
            elif height > width:
                target_height = 4096
                target_width = roundint(4096 * ratio, 8)
            else:
                target_width, target_height = 4096, 4096

        XL = isinstance(clip.cond_stage_model, comfy.sdxl_clip.SDXLClipModel)
        REF = isinstance(clip.cond_stage_model, comfy.sdxl_clip.SDXLRefinerClipModel)

        if DEBUG: print(f"\nText:\n{self.text}")
        if DEBUG: print(f"\nXL: {XL}\nREF: {REF}\nW / H: {width} / {height}\nTW / TH: {target_width} / {target_height}\nASC: {self.refiner_asc}\n")

        if XL: return nodes_xl.CLIPTextEncodeSDXL.encode(
            None,
            clip,
            width,
            height,
            0,
            0,
            target_width,
            target_height,
            self.text,
            self.text,
        )[0]
        elif REF: return nodes_xl.CLIPTextEncodeSDXLRefiner.encode(
            None,
            clip,
            self.refiner_asc,
            width, # should these be target?
            height,
            self.text,
        )[0]
        else: return nodes.CLIPTextEncode.encode(
            None,
            clip,
            self.text,
        )[0]
        # }}}

class CondAnd(CondStage):
    def process(self, latent, clip, other):
        cond = self._encode(latent, clip)
        return nodes.ConditioningCombine.combine(None, other, cond)[0]

class CondBreak(CondStage):
    def process(self, latent, clip, other):
        cond = self._encode(latent, clip)
        return nodes.ConditioningConcat.concat(None, other, cond)[0]

class BSZPrincipledScale:
    #{{{
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "method": (list(METHODS_LATENT.keys()) + list(METHODS_PIXEL.keys()) + list(METHODS_MODEL.keys()),),
                "width": ("INT", {"default": 1024, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "scale"
    CATEGORY = "beinsezii/image"

    def scale(self, vae, latent, method, width, height):
        latent_height, latent_width = latent["samples"].size()[2:4]
        latent_height *= 8
        latent_width *= 8

        if latent_width != width or latent_height != height:
            if method in METHODS_LATENT:
                latent = nodes.LatentUpscale.upscale(None, latent, METHODS_LATENT[method], width, height, "disabled")[0]
            else:
                decoder = nodes.VAEDecode()
                pixels = decoder.decode(vae, latent)[0]
                del decoder
                if method in METHODS_PIXEL:
                    pixels = nodes.ImageScale.upscale(None, pixels, METHODS_PIXEL[method], width, height, "disabled")[0]
                elif method in METHODS_MODEL:
                    scale_model = nodes_scale.UpscaleModelLoader.load_model(None, METHODS_MODEL[method])[0]
                    pixels = nodes_scale.ImageUpscaleWithModel.upscale(None, scale_model, pixels)[0]
                    del scale_model
                    pixels = nodes.ImageScale.upscale(None, pixels, 'bicubic', width, height, "disabled")[0]
                else:
                    raise ValueError("Unreachable!")

                encoder = nodes.VAEEncode()
                latent = encoder.encode(vae, pixels)[0]
                del pixels, encoder
        return (latent,)
    # }}}

class BSZPrincipledConditioning:
    # {{{
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True}),
                "xl_target": (["1k", "full", "4k"],),
                "refiner_asc": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "beinsezii/conditioning"
    FUNCTION = "encode"
    def encode(self, latent, clip, text: str, xl_target="1k", refiner_asc=6.0):
        return (CondStage(text, xl_target, refiner_asc).process(latent, clip, None),)
    # }}}

class BSZPrincipledSampler:
    # {{{
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "refiner_model": ("MODEL",),
                "refiner_clip": ("CLIP",),
            },
            "required": {
                "base_model": ("MODEL",),
                "base_clip": ("CLIP",),
                "latent": ("LATENT",),
                "positive_prompt": ("STRING", {
                    "multiline": True,
                    "default": "analogue photograph of a kitten"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "blurry, cropped, text"
                }),
                "steps": ("INT", {"default": 30, "min": 0, "max": 10000}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "refiner_amount": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "refiner_asc_pos": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "refiner_asc_neg": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "ddim"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = (
        "LATENT",
        "STRING", # neg
        "STRING", # pos
        "INT", # steps
        "FLOAT", # den
        "FLOAT", # cfg
        "FLOAT", # ref
        "FLOAT", # ascp
        "FLOAT", # ascn
        comfy.samplers.KSampler.SAMPLERS, # sampler
        comfy.samplers.KSampler.SCHEDULERS, # scheduler
        "INT", # seed
    )
    RETURN_NAMES = (
        "latent",
        "positive_prompt",
        "negative_prompt",
        "steps",
        "denoise",
        "cfg",
        "refiner_amount",
        "refiner_asc_pos",
        "refiner_asc_neg",
        "sampler",
        "scheduler",
        "seed",
    )

    FUNCTION = "principled"

    #OUTPUT_NODE = False

    CATEGORY = "beinsezii/sampling"

    def principled(
        self,
        base_model,
        base_clip,
        latent,
        positive_prompt: str,
        negative_prompt: str,
        steps: int,
        denoise: float,
        cfg: float,
        refiner_amount: float,
        refiner_asc_pos: float,
        refiner_asc_neg: float,
        sampler,
        scheduler,
        seed: int,
        refiner_model=None,
        refiner_clip=None,
    ):
    #{{{
        # hot patch function before sampling
        comfy.sample.prepare_noise = _prepare_noise

        # disable refiner if not provided
        if refiner_model is None and refiner_clip is None:
            refiner_amount = 0
            scale_refiner_amount = 0
        elif refiner_model is not None and refiner_clip is not None:
            pass
        else:
            raise Exception("You must provide both refiner model and refiner clip to use the refiner")

        # steps skipped by img2img are effectively base steps as far
        # as the refiner is concerned
        adjusted_refiner_amount = min(1, refiner_amount / max(denoise, 0.00001))
        base_start = round(steps - steps * denoise)
        base_end = round((steps - base_start) * ( 1 - adjusted_refiner_amount) + base_start)

        base_run = False

        if DEBUG: print(f"Sampling start - seed: {seed} cfg: {cfg}\npositive: {positive_prompt}\nnegative: {negative_prompt}")
        if base_start < base_end:
            if DEBUG: print(f"Running Base - total: {steps} start: {base_start} end: {base_end}")
            try:
                latent = nodes.common_ksampler(
                    base_model,
                    seed,
                    steps,
                    cfg,
                    sampler,
                    scheduler,
                    BSZPrincipledConditioning.encode(None, latent, base_clip, positive_prompt)[0],
                    BSZPrincipledConditioning.encode(None, latent, base_clip, negative_prompt)[0],
                    latent,
                    start_step=base_start,
                    last_step=None if base_end == steps else base_end,
                    force_full_denoise=False if base_end < steps else True,
                )[0]
            except Exception as e:
                # restore patched function if canceled before forwarding err
                comfy.sample.prepare_noise = OLD_PREPARE_NOISE
                raise e
            base_run = True

        if base_end < steps:
            if DEBUG: print(f"Running Refiner - total: {steps} start: {base_end} ascore: +{refiner_asc_pos} -{refiner_asc_neg}")
            try:
                latent = nodes.common_ksampler(
                    refiner_model,
                    seed,
                    steps,
                    cfg,
                    sampler,
                    scheduler,
                    BSZPrincipledConditioning.encode(None, latent, refiner_clip, positive_prompt, refiner_asc=refiner_asc_pos)[0],
                    BSZPrincipledConditioning.encode(None, latent, refiner_clip, negative_prompt, refiner_asc=refiner_asc_neg)[0],
                    latent,
                    start_step=base_end,
                    force_full_denoise=True,
                    disable_noise=base_run
                )[0]
            except Exception as e:
                # restore patched function if canceled before forwarding err
                comfy.sample.prepare_noise = OLD_PREPARE_NOISE
                raise e

        # restore patched function on successfuly finish
        comfy.sample.prepare_noise = OLD_PREPARE_NOISE

        return (
            latent,
            positive_prompt,
            negative_prompt,
            steps,
            denoise,
            cfg,
            refiner_amount,
            refiner_asc_pos,
            refiner_asc_neg,
            sampler,
            scheduler,
            seed,
        )
    #}}}
    #}}}

NODE_CLASS_MAPPINGS = {
    "BSZPrincipledSampler": BSZPrincipledSampler,
    "BSZPrincipledScale": BSZPrincipledScale,
    "BSZPrincipledConditioning": BSZPrincipledConditioning,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BSZPrincipledSampler": "BSZ Principled Sampler",
    "BSZPrincipledScale": "BSZ Principled Scale",
    "BSZPrincipledConditioning": "BSZ Principled Conditioning",
}
