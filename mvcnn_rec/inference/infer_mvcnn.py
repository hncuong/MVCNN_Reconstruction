import random
from pathlib import Path

import torch
import torch.nn as nn

from mvcnn_rec.model.mvcnn import MVCNN
from mvcnn_rec.data.shapenet import ShapeNetMultiview

class InferenceHandlerMVCNN:
    """Utility for inference using trained MVCNN network"""

    def __init__(self, ckpt) -> None:
        """
        :param ckpt: checkpoint path to weights of the trained network
        """
        self.model = MVCNN(ShapeNetMultiview.num_classes)
        self.model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        self.model.eval()
    
    def infer_single(self, images):
        """
        Infer class of the shape given its multi-views representation
        :param images: multi-views images of shape N_views x 3 x 224 x 224
        :return: class category name for the voxels, as predicted by the model
        """
        input_tensor = torch.unsqueeze(images, 1) # N_views x 1 x 3 x 224 x 224

        # TODO: Predict class
        prediction = self.model(input_tensor)
        predicted_label = torch.argmax(prediction, dim=1)

        class_id = predicted_label.item()
        class_name = ShapeNetMultiview.id_class_mapping[class_id]
        return class_name
    




