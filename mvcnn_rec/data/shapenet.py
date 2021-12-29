from pathlib import Path
import json
import os
import cv2
import numpy as np
import torch
from config import cfg
from torchvision import transforms
from mvcnn_rec.data.binvox_rw import read_as_3d_array
from PIL import Image

class ShapeNetMultiview(torch.utils.data.Dataset):
    """Class for loading multiview images, class, and voxel of object
    """
    def __init__(self, split):
        super().__init__()
        assert split in ['train', 'test', 'val']
        self.items = []
        self.classes = {}
        self.class_ids = {
            "aeroplane" : 0,
            "bench" : 1,
            "cabinet": 2,
            "car": 3,
            "chair": 4,
            "display": 5,
            "lamp": 6,
            "speaker": 7,
            "rifle": 8,
            "sofa": 9,
            "table": 10,
            "telephone": 11,
            "watercraft": 12
        }
        with open(cfg.DATASETS.SHAPENET.TAXONOMY_FILE_PATH, encoding='utf-8') as file:
            content = file.read()

        print(cfg.DATASETS.SHAPENET.TAXONOMY_FILE_PATH)
        dataset_taxonomy = json.loads(content)
        #print(dataset_taxonomy)

        for taxonomy in dataset_taxonomy:
            self.classes[taxonomy['taxonomy_id']] = taxonomy['taxonomy_name']
            #print('"%s": "%s"' % (taxonomy['taxonomy_name'], taxonomy['taxonomy_id']))
            for item in taxonomy[split]:
                self.items.append("%s/%s" % (taxonomy['taxonomy_id'], item))

    def __getitem__(self, index):
        item = self.items[index]
        print(f"item: {item}")
        item_class = item.split('/')[0]
        images = ShapeNetMultiview.get_images(item)
        voxel = ShapeNetMultiview.get_voxel(item)
        return {
            "name": item,
            "item": images,
            "voxel": voxel[np.newaxis, :, :, :],
            "label": self.class_ids[self.classes[item_class]]
        }

    def __len__(self):
        return len(self.items)

    @staticmethod
    def get_voxel(item):
        voxel_path = cfg.DATASETS.SHAPENET.VOXEL_PATH % (item.split('/')[0], item.split('/')[1])
        with open(voxel_path, "rb") as voxel_file:
            voxels = read_as_3d_array(voxel_file).astype(np.float32)
        return voxels

    @staticmethod
    def get_images(item):
        preprocessed_images = []
        preprocess = transforms.Compose([
            transforms.Resize(222),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        for i in range(0, 23):
            img_path = cfg.DATASETS.SHAPENET.RENDERING_PATH % (item.split('/')[0], item.split('/')[1], i)
            #print(f"open img_path: {img_path}")
            #image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            #cv2.imshow('image', image)
            #cv2.waitKey(0)
            #rendering_image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
            #rendering_image = preprocess(cv2.resize(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), (222, 222),
            #                            interpolation=cv2.INTER_NEAREST).astype(np.float32) )
            #rendering_image = preprocess(rendering_image)
            #print(rendering_image.shape)
            input_image = Image.open(img_path)
            input_tensor = preprocess(input_image)
            preprocessed_images.append(input_tensor)
        return np.asarray(preprocessed_images)


    @staticmethod
    def move_batch_to_device(batch, device):
        batch["item"] = batch['item'].to(device, torch.float32)
        batch['voxel'] = batch['voxel'].to(device, torch.float32)
        batch['label'] = batch['label'].to(device, torch.float32)
