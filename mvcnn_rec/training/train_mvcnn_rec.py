from pathlib import Path

import torch

from mvcnn_rec.model.mvcnn_rec import MVCNNReconstruction
from mvcnn_rec.data.shapenet import ShapeNetMultiview

def train(model, latent_vectors, train_dataloader, device, config):
    raise NotImplementedError

def main(config):
    raise NotImplementedError