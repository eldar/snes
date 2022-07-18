import os
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

import neptune.new as neptune
from neptune_token import NEPTUNE_API_TOKEN


def setup(cfg):
    CONNECTION_MODE = "debug" if cfg.run.debug else "async"
    run = neptune.init(
        project=f"{cfg.logging.entity}/{cfg.logging.project}",
        api_token=NEPTUNE_API_TOKEN,
        name=cfg.config.exp_name,
        mode=CONNECTION_MODE)
    return run


def setup_logging_dir():
    tmp_dir = tempfile._get_default_tempdir()
    neptune_dir = Path(tmp_dir).joinpath(".neptune")
    neptune_dir.mkdir(exist_ok=True)
    target = Path(".neptune")
    if target.exists():
        os.unlink(target)
    os.symlink(neptune_dir, ".neptune")


class NeptuneLogger:
    def __init__(self, cfg):
        cfg_dict = OmegaConf.to_container(cfg)
        del cfg_dict["config"]
        setup_logging_dir()
        self.run = setup(cfg)
        self.run['cfg'] = cfg_dict
        self.run["exp"] = cfg.config.exp_name

        if "SLURM_JOB_ID" in os.environ:
            SLURM_ID = os.environ['SLURM_JOB_ID']
            self.run["SLURM"] = SLURM_ID
            print(f"SLURM job ID: {SLURM_ID}")

    def log(self, values, step):
        for key, value in values.items():
            self.run[key].log(value, step=step)
    
    def log3d(self, kv, step):
        pass

    def upload_file(self, key, filename):
        self.run[key].upload(neptune.types.File(filename))
    
    def upload_image(self, key, image):
        self.run[key].upload(neptune.types.File.as_image(image))
