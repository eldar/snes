import importlib


def factory(cfg):
    m = importlib.import_module(f"models.{cfg.model.renderer.renderer}")
    return getattr(m, "Renderer")
