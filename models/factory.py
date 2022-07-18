import torch

def dataset_factory(cfg):
    typ = cfg.type
    if typ == "dtu":
        from models.dataset import Dataset as Dataset_DTU
        return Dataset_DTU
    elif typ == "co3d":
        from models.dataset_co3d import Dataset as Dataset_Co3D
        return Dataset_Co3D


def load_datasets(cfg,
                  dataset_device = torch.device('cpu')):
    dataset_cfg = cfg.dataset
    dataset_cls = dataset_factory(dataset_cfg)
    dataset = dataset_cls(dataset_cfg, device=dataset_device)
    if dataset_cfg.trainval_split:
        val_dataset = dataset_cls(dataset_cfg,
                                  split="val",
                                  other=dataset,
                                  device=dataset_device)
    else:
        val_dataset = None
    return dataset, val_dataset
