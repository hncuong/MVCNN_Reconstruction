import random
from pathlib import Path

import torch
import torch.nn as nn

from mvcnn_rec.model.mvcnn_rec import MVCNNReconstruction
from mvcnn_rec.data.shapenet import ShapeNetMultiview

class InferenceHandlerMVCNNReconstruction:
    def __init__(self, ckpt) -> None:
        """
        :param ckpt: checkpoint path to weights of the trained network
        """
        self.model = MVCNNReconstruction(ShapeNetMultiview.num_classes)
        self.model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        self.model.eval()
    
    def infer_single(self, images, voxel_thresh=0.5):
        """
        Infer class and voxel of the shape given its multi-views representation
        :param images: multi-views images of shape N_views x 3 x 224 x 224
        :param voxel_thresh: threshold for voxel in range (0, 1)
        :return: class category name for the voxels, as predicted by the model
        """
        input_data = torch.unsqueeze(images, 1) # N_views x 1 x 3 x 224 x 224

        with torch.no_grad():
            # prediction = model(input_data)
            pred_voxels, pred_class = self.model(input_data)
        # Class
        predicted_label = torch.argmax(pred_class, dim=1)
        class_id = predicted_label.item()
        class_name = ShapeNetMultiview.id_class_mapping[class_id]

        # Thresh hold voxel
        predited_voxel = (pred_voxels.squeeze(0) > voxel_thresh).numpy()
        return class_name, predited_voxel



