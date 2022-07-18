import os
from pathlib import Path
import functools
from typing import Any, Callable, Optional
import inspect

from omegaconf import DictConfig, OmegaConf

s = """
config:
  exp_path: ""
  exp_name: ""
  file: "config.yaml"
"""
_base_conf = OmegaConf.create(s)
_original_cwd = None


def get_original_cwd():
    return _original_cwd


def construct_config(default_config, override_config):
    cfg_default = OmegaConf.load(os.path.join(os.getcwd(), default_config))
    cfg = OmegaConf.merge(_base_conf, cfg_default, override_config)

    exp_dir = Path(cfg.config.exp_path, cfg.config.exp_name)
    assert cfg.config.exp_name, "Please specify experiment name"
    print("Experiment name:", cfg.config.exp_name, "\n")
    if not exp_dir.exists():
        exp_dir.mkdir()
        os.chmod(exp_dir, 0o777)

    global _original_cwd
    _original_cwd = os.getcwd()
    os.chdir(exp_dir)

    if os.path.isfile(cfg.config.file):
        exp_cfg = OmegaConf.load(cfg.config.file)
        cfg = OmegaConf.merge(_base_conf, cfg_default, exp_cfg, override_config)
    elif os.path.isfile(".full_config.yaml"):
        exp_cfg = OmegaConf.load(".full_config.yaml")
        cfg = OmegaConf.merge(_base_conf, cfg_default, exp_cfg, override_config)
    
    return cfg


def construct_config_from_cli(default_config):
    cfg_cli = OmegaConf.from_cli()
    full_config = OmegaConf.merge(_base_conf, cfg_cli)
    if os.path.isfile(full_config.config.file):
        exp_cfg = OmegaConf.load(cfg_cli.config.file)
        cfg = OmegaConf.merge(exp_cfg, cfg_cli)
    else:
        cfg = cfg_cli
    return construct_config(default_config, cfg)


def main(
    default_config: Optional[str] = None
):
    def main_decorator(task_function):
        @functools.wraps(task_function)
        def decorated_main():
            cfg = construct_config_from_cli(default_config)           
            full_config_file = ".full_config.yaml"
            if not os.path.isfile(full_config_file):
                OmegaConf.save(cfg, full_config_file)
            task_function(cfg)

        return decorated_main

    return main_decorator
