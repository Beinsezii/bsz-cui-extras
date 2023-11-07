import nodes
import comfy

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

NODE_CLASS_MAPPINGS = {
    "BSZInjectionKSampler": BSZInjectionKSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSZInjectionKSampler": "BSZ Injection KSampler",
}

