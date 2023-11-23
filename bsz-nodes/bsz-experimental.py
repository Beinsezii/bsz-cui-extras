import nodes
import comfy
import torch
import numpy
import PIL

class BSZInjectionKSampler:
    # {{{
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.5, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "injection": ("LATENT", ),
                    "time": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step":0.01}),
                    "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step":0.05}),
                     }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "beinsezii/experimental"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, injection, time, strength):
        assert latent_image['samples'].shape == injection['samples'].shape
        split = round(steps * time)
        latent_image = nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_step=0, last_step=split, disable_noise=False, force_full_denoise=False, denoise=denoise)[0]
        latent_image['samples'] += injection['samples'].mul(strength)
        return nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_step=split, last_step=None, disable_noise=True, force_full_denoise=True, denoise=denoise)

    #}}}

pil_modes = {
        "PIL_Nearest": PIL.Image.Resampling.NEAREST,
        "PIL_Box": PIL.Image.Resampling.BOX,
        "PIL_Bilinear": PIL.Image.Resampling.BILINEAR,
        "PIL_Hamming": PIL.Image.Resampling.HAMMING,
        "PIL_Bicubic": PIL.Image.Resampling.BICUBIC,
        "PIL_Lanczos": PIL.Image.Resampling.LANCZOS,
}
class BSZStrangeResample:
    #{{{
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "method": (list(pil_modes.keys()),),
                "width": ("INT", {"default": 1024, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "resample"
    CATEGORY = "beinsezii/experimental"

    def resample(self, latent, method, width, height):
        b, c, h, w = latent["samples"].shape
        result = latent.copy()
        if method in list(pil_modes.keys()):
            tensors = []
            for batch in latent["samples"]:
                channels = []
                for channel in batch:
                    tensor = channel.to(torch.float32).numpy().copy()
                    img: PIL.Image.Image = PIL.Image.fromarray(tensor, 'F').resize((width // 8, height // 8), resample=pil_modes[method])
                    tensor = torch.from_numpy(numpy.asarray(img))
                    shape = [1] + list(tensor.shape)
                    channels.append(tensor.reshape(shape))
                tensor = torch.cat(channels)
                shape = [1] + list(tensor.shape)
                tensors.append(tensor.reshape(shape))
            result['samples'] = torch.cat(tensors)
        return (result,)
    # }}}

NODE_CLASS_MAPPINGS = {
    "BSZInjectionKSampler": BSZInjectionKSampler,
    "BSZStrangeResample": BSZStrangeResample,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSZInjectionKSampler": "BSZ Injection KSampler",
    "BSZStrangeResample": "BSZ Strange Resample",
}

