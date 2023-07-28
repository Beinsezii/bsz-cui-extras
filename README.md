# bsz-cui-extras
Addons for [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

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

### bsz-principled-sdxl.py
All-in-one solution for SDXL text2img, img2img and scaling/hi res fix. Essentially the sdxl and sdxl-upscale workflows both in one node. Do note that while this node shouldn't be any slower than the regular workflow, due to ComfyUI caching latent results **per-node**, even changing just a refiner setting on this node will result in sampling starting over from the first base pass. There are at least some minimal internal optimizations to skip passes that aren't needed.

Scaling works by running an initial pass before scaling to `target` size and running another

Input fields
  - `base_model` : Model from base checkpoint
  - `base_clip` : CLIP from base checkpoint
  - `refiner_model` : Model from refiner checkpoint
  - `refiner_clip` : CLIP from refiner checkpoint
  - `latent_image` : Latent image to start from. Optional, useful mostly for img2img
  - `pixel_scale_vae` : VAE used for pixel scaling methods. Only needed if they're being used
  - `positive_prompt_G` : Positive prompt for base CLIP G and refiner
  - `positive_prompt_L` : Positive prompt for base CLIP L. Typically viewed as "supporting terms" for the main G prompt. Setting both L and G to the same value is completely valid
  - `negative_prompt` : Negative prompt
  - `steps` : Total steps
  - `denoise` : Denoise amount for img 2 img
  - `cfg` : CFG scale
  - `refiner_amount` : Refiner to base ratio
  - `refiner_ascore_positive` : Refiner aesthetic score for positive prompt
  - `refiner_ascore_negative` : Refiner aesthetic score for negative prompt
  - `width` : CLIP input width in pixels. If no `latent_image` is provided, will generate one with this size
  - `height` : CLIP input height in pixels. If no `latent_image` is provided, will generate one with this size
  - `target_width` : CLIP target width in pixels. If `scale_method` is enabled, image will be resized to this
  - `target_height` : CLIP target height in pixels. If `scale_method` is enabled, image will be resized to this
  - `sampler` : Sampler
  - `scheduler` : Scheduler
  - `scale_method` : If set, will scale image to match target sizes using the provided algorithm
  - `scale_denoise` : Amount to denoise after scale
  - `scale_initial_steps` : Total target steps for pre-scale pass
  - `scale_initial_cutoff` : End the initial pre-scale pass early to save time. Only useful for latent scaling, pixel scaling should be 1.0
  - `seed` : Seedy.

Recommended settings for various workflows...

  - Text2Image: default
  - Text2Image w/latent upscale:
    - `scale_method`:`latent bilinear`
  - Text2Image w/pixel upscale:
    - `scale_method` : `pixel cubic`
    - `scale_denoise` : `0.1`
    - `scale_initial_cutoff` : `1.0`
  - Img2Img w/upscale:
    - Same as text2img upscaling
    - `scale_initial_steps` : `0`

## Workflows

### sdxl.json
Personal flair of the SDXL "partial diffusion" workflow. Minimalist node setup with defaults balanced approach to speed/quality

#### Dependencies
  - `bsz-auto-hires.py` : While this workflow doesn't actually perform any upscaling, it still uses the `BSZAutoHiresCombined` node for quick aspect ratio changing and easy CLIP detail target adjustments

### sdxl-upscale.json
Personal flair of the SDXL "partial diffusion" workflow with added "High res fix". Slightly prioritizes speed as far as upscaling is concerned.

#### Dependencies
  - `bsz-auto-hires.py` : Workflow is painful without it.

### sdxl-principled.json
Demonstration of the bsz-principled-sdxl node

#### Dependencies
  - `bsz-auto-hires.py` : Principled can use hi res sizes
  - `bsz-principled-sdxl.py` : Yes.


## F.A.Q.
Question|Answer
---|---
Why is there a separate VAE loader instead of using the VAE directly from the main checkpoint?|I personally find it desireable to have the VAE decoupled from the checkpoint so you can change it without re-baking the models. If this isn't desirable to you yourself, simply remove the Load VAE node and reconnect the traces into the main Load Checkpoint node instead.
Why are the KSAmpler nodes so long?|To show live previews of each stage. I strongly recommend you do the same by launching ComfyUI with `--preview-method latent2rgb` or similar.
Why is *this* setting the default instead of *that* setting?|It just happens to look better on my benchmark images. If you think it's objectively wrong, open an issue with a compelling case on why it should be changed.
You should add the refiner 1 step detail trick|No. That "trick" really just causes the refiner to interpret latent noise as "details" it should refine, which hurts the overall image quality. If you render an image and really really think it needs it, just load the most recent history item and adjust the refiner steps as needed. ComfyUI caches the previous latents so you won't have to re-render the whole image, just the part that changed.
