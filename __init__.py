"""
Forwards the contents of ./bsz-nodes/ for reading when the whole repo is cloned into ComfyUI's /custom_nodes/ folder

Should make it compatible with https://github.com/ltdrdata/ComfyUI-Manager
"""

import importlib
loader = importlib.find_loader('bsz-nodes', __path__)
mod = loader.load_module()

NODE_CLASS_MAPPINGS = mod.NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = mod.NODE_DISPLAY_NAME_MAPPINGS
