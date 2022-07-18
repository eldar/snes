from omegaconf import OmegaConf
import os
from pathlib import Path
from util.config import construct_config
from exp_runner import Runner


def get_models_list(filename="lists/valid_ids.txt"):
    with open(filename) as f:
        lines = f.readlines()
    instance_ids = [l.rstrip() for l in lines]
    return instance_ids


def iterate_exp(
    exp_root,
    instance_list,
    init_cfg=dict(),
    cfg_file=None,
    with_runner=False):
    exp_root = Path(exp_root)
    if isinstance(instance_list, list):
        ids = instance_list
    else:
        ids = get_models_list(instance_list)
        # ids = [id for id in ids if Path("exp", exp_root, id, "checkpoints", "ckpt_300000.pth").exists()]
    for instance in ids:
        code_dir = Path(os.getcwd())

        cfg_dict = {
            "config" : {"exp_name": str(exp_root.joinpath(instance))},
            "dataset": {"instance": instance},
            "mode": "else",
            "is_continue": True
        }
        if cfg_file is not None:
            cfg_dict["config"]["file"] = cfg_file
        cfg = OmegaConf.create(cfg_dict)
        cfg = OmegaConf.merge(cfg, OmegaConf.create(init_cfg))
        cfg = construct_config("config/config.yaml", cfg)

        if with_runner:
            yield Runner(cfg)
        else:
            yield cfg
        
        os.chdir(code_dir)