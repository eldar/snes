import os
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, cfg):
        self.base_exp_dir = os.getcwd()
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))

    def log(self, kv, step):
        for key, value in kv.items():
            self.writer.add_scalar(key, value, step)

    def upload_image(self, name, image):
        self.writer.add_image(name, image, dataformats="HWC")

    def upload_file(self, name, file):
        pass