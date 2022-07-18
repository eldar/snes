def create_logger(cfg):
    if cfg.logging.backend == "wandb":
        from .wandb_logger import WANDBLogger
        return WANDBLogger(cfg)
    elif cfg.logging.backend == "neptune":
        from .neptune_logger import NeptuneLogger
        return NeptuneLogger(cfg)
    elif cfg.logging.backend == "tensorboard":
        from .tensorboard_logger import TensorBoardLogger
        return TensorBoardLogger(cfg)
    else:
        from .dummy_logger import DummyLogger
        return DummyLogger(cfg)