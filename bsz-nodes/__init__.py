import importlib
import pkgutil

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for finder in pkgutil.iter_modules(__path__):
    try:
        spec = finder.module_finder.find_spec(finder.name)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        NODE_CLASS_MAPPINGS |= mod.NODE_CLASS_MAPPINGS
        NODE_DISPLAY_NAME_MAPPINGS |= mod.NODE_DISPLAY_NAME_MAPPINGS
    except Exception:
        print(f"[ERROR] bsz-cui-extras: Failed to load '{finder.name}' module")
