# bsz-cui-extras
Addons for ComfyUI

## Custom Nodes

### bsz-auto-hires.py
Contains 3 nodes each with a different means to the same end result.
These nodes are designed to automatically calculate the appropriate latent sizes when performing a "Hi Res Fix" style workflow.

#### Common:
 - Input
  - `base_model_res` : Resolution of base model being used. SD 1.5 ≅ 512, SD 2.1 ≅ 768, SDXL ≅ 1024
 - Output
  - `Lo Res Width` : Width intended to be used for first/low res pass
  - `Lo Res Height` : Height intended to be used for first/low res pass
  - `Hi Res Width` : Width intended to be used for final/high res pass
  - `Hi Res Height` : Height intended to be used for final/high res pass

#### BSZAutoHires:
 - Input
  - `desired_width` : Width in pixels for final/high res pass.
  - `desired_height` : Height in pixels for final/high res pass.

#### BSZAutoHiresAspect:
 - Input
  - `desired_aspect_x` : Horizontal aspect.
  - `desired_aspect_Y` : Vertical aspect.
  - `scale` : Hi Res horizontal and vertical scale over Lo Res sizes. Note that because this scales both axes, a scale of `2.0` will actually quadruple the amount of pixels in an image, so use with care.

#### BSZAutoHiresCombined:
A unique node that functions both as BSZAutoHires and BSZAutoHiresAspect with a convenient toggle
 - Input
  - `use_aspect_scale_instead` : Use aspect & scale inputs instead of desired width/height inputs

## Workflows

### sdxl.json
Personal flair of the SDXL "partial diffusion" workflow. Minimalist node setup with defaults balanced approach to speed/quality

#### Dependencies
 - `bsz-auto-hires.py` : While this workflow doesn't actually perform any upscaling, it still uses the `BSZAutoHiresCombined` node for quick aspect ratio changing and easy CLIP detail target adjustments

### sdxl.json
Personal flair of the SDXL "partial diffusion" workflow with added "High res fix". Slightly prioritizes speed as far as upscaling is concerned.

#### Dependencies
 - `bsz-auto-hires.py` : Workflow is painful without it.

## F.A.Q.
Question|Answer
---|---
Why is there a separate VAE loader instead of using the VAE directly from the main checkpoint?|I personally find it desireable to have the VAE decoupled from the checkpoint so you can change it without re-baking the models. If this isn't desirable to you yourself, simply remove the Load VAE node and reconnect the traces into the main Load Checkpoint node instead.
Why are the KSAmpler nodes so long?|To show live previews of each stage. I strongly recommend you do the same by launching ComfyUI with `--preview-method latent2rgb` or similar.
Why is *this* setting the default instead of *that* setting?|It just happens to look better on my benchmark images. If you think it's objectively wrong, open an issue with a compelling case on why it should be changed.
You should add the refiner 1 step detail trick|No. That "trick" really just causes the refiner to interpret latent noise as "details" it should refine, which hurts the overall image quality. If you render an image and really really think it needs it, just load the most recent history item and adjust the refiner steps as needed. ComfyUI caches the previous latents so you won't have to re-render the whole image, just the part that changed.
