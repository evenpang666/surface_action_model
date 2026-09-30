"""Load the pinned repository Transformers extensions without editing site-packages."""
import importlib
import importlib.util
from importlib.metadata import version
from pathlib import Path
import sys

_installed = False


def install_repository_extensions():
    global _installed
    if _installed:
        return
    if version("transformers") != "4.53.2":
        raise RuntimeError("Surface Action Model requires transformers==4.53.2 for the repository extensions")
    root = Path(__file__).parent / "transformers_replace"
    specs = (
        ("models.gemma.configuration_gemma", ("GemmaConfig",)),
        ("models.gemma.modeling_gemma", ("GemmaModel", "GemmaForCausalLM", "GemmaPreTrainedModel")),
        ("models.siglip.modeling_siglip", ("SiglipModel", "SiglipVisionModel", "SiglipTextModel", "SiglipPreTrainedModel")),
        ("models.siglip.check", ()),
        ("models.paligemma.modeling_paligemma", ("PaliGemmaForConditionalGeneration", "PaliGemmaPreTrainedModel")),
    )
    transformers = importlib.import_module("transformers")
    for relative, exports in specs:
        name = "transformers." + relative
        parent_name, child = name.rsplit(".", 1)
        parent = importlib.import_module(parent_name)
        spec = importlib.util.spec_from_file_location(name, root.joinpath(*relative.split(".")).with_suffix(".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        setattr(parent, child, module)
        for export in exports:
            if hasattr(module, export):
                setattr(parent, export, getattr(module, export))
                setattr(transformers, export, getattr(module, export))
    _installed = True
