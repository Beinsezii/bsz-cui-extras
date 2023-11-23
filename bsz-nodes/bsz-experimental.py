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
                "method": (['slurry2'] + list(pil_modes.keys()),),
                "width": ("INT", {"default": 1024, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "bleed": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "resample"
    CATEGORY = "beinsezii/experimental"

    def bleed(self, t, bleed):
        result = torch.empty([4])
        result[0] = t[1] * bleed + t[3] * bleed
        result[1] = t[0] * bleed + t[2] * bleed
        result[2] = t[3] * bleed + t[1] * bleed
        result[3] = t[2] * bleed + t[0] * bleed
        result.div_(2)
        return result

    def resample(self, latent, method, width, height, bleed):
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
        if method == 'slurry2':
            bleed *= (1 / 16)
            tensor = torch.zeros([b, w*2, h*2, c])
            old = result['samples'].permute(0, 3, 2, 1)
            for bn, batch in enumerate(tensor):
                for xn, x in enumerate(batch):
                    for yn, y in enumerate(x):
                        xo, yo = xn // 2, yn // 2
                        nearest = old[bn][xo][yo]
                        if xn % 2 == 1 or yn % 2 == 1:
                            end = nearest.clone()
                            bleeds = []
                            n = 1

                            if xo < w-1 and xn % 2:
                                t = old[bn][xo+1][yo]
                                end += t
                                bleeds.append(self.bleed(t, bleed))
                                n += 1
                            if xo > 0 and xn % 2:
                                t = old[bn][xo-1][yo]
                                end += t
                                bleeds.append(self.bleed(t, bleed))
                                n += 1
                            if yo > 0 and yn % 2:
                                t = old[bn][xo][yo-1]
                                end += t
                                bleeds.append(self.bleed(t, bleed))
                                n += 1
                            if yo < h-1 and yn % 2:
                                t = old[bn][xo][yo+1]
                                end += t
                                bleeds.append(self.bleed(t, bleed))
                                n += 1

                            end.div_(n)

                            for b in bleeds:
                                end += b
                            end.div_(1 + (bleed / (1 + bleed)) * n)

                            tensor[bn][xn][yn] = end
                        else:
                            tensor[bn][xn][yn] = nearest
            result['samples'] = tensor.permute(0, 3, 2, 1)
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

