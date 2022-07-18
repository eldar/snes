import os
import tempfile
import wandb
from omegaconf import OmegaConf


class WANDBLogger:
    def __init__(self, cfg):
        cfg_dict = OmegaConf.to_container(cfg)
        del cfg_dict["config"]
        wandb.init(
            project=cfg.logging.project,
            entity=cfg.logging.entity,
            config=cfg_dict,
            name=cfg.config.exp_name)

    def log(self, values, step):
        wandb.log(values, step=step)
    
    def log3d(self, kv, step):
        for key, value in kv.items():
            mesh = value
            tmp_name = next(tempfile._get_candidate_names())
            tmp_dir = tempfile._get_default_tempdir()
            tmp_mesh_file = os.path.join(tmp_dir, f"{tmp_name}.obj")
            mesh.export(tmp_mesh_file)
            wandb.log({key: wandb.Object3D(open(tmp_mesh_file))}, step=step)

