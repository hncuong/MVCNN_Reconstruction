from pathlib import Path
import json

import numpy as np
import torch


class ShapeNetMultiview(torch.utils.data.Dataset):
    """Class for loading multiview images, class, and voxel of object
    """

    def __init__(self, split):
        super().__init__()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @staticmethod
    def move_batch_to_device(batch, device):
        raise NotImplementedError
