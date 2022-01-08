import random
from pathlib import Path

import torch
import torch.nn as nn

from mvcnn_rec.model.mvcnn_rec import MVCNNReconstruction
from mvcnn_rec.data.shapenet import ShapeNetMultiview

class InferenceHandlerMVCNNReconstruction:
    def __init__(self, path):
        self.model = MVCNNReconstruction()
        self.model.load_state_dict(torch.load(path, map_location='cpu'))
        self.model.eval()
        self.class_ids = [
            "aeroplane",
            "bench",
            "cabinet",
            "car",
            "chair",
            "display",
            "lamp",
            "speaker",
            "rifle",
            "sofa",
            "table",
            "telephone",
            "watercraft"
        ]

    def infer_single(self, input_images):
        with torch.no_grad():
            infer_class, infer_voxel = self.model(input_images)
            class_name = self.class_ids[torch.argmax(infer_class).item()]
        return class_name, infer_voxel

class InferenceHandlerMVCNNClassification:
    def __init__(self, path):
        pass



