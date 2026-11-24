from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[3] / "examples" / "vla_autotune"


def example(filename):
    spec = spec_from_file_location("_author_" + Path(filename).stem, EXAMPLES / filename)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prompt(request):
    text = request.prompt if request.prompt is not None else request.observation.get('prompt')
    if hasattr(text, 'item'):
        text = text.item()
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Author request requires an explicit nonempty prompt')
    return text


def _cameras(request, names):
    images = request.observation.get('images')
    if isinstance(images, dict):
        images = [images[name] for name in names]
    elif images is None:
        images = [request.observation[name] for name in names]
    if len(images) != len(names):
        raise ValueError('Author request must supply exactly the configured cameras in order')
    return images
