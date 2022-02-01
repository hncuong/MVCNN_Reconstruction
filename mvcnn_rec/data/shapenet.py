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
    """ 
    Class for loading multiview images, class, and voxel of object
    """
    num_classes = 13
    class_ids = {
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
    id_class_mapping = {idx: class_name for class_name, idx in class_ids.items()}

    def __init__(self, split, total_views=24, num_views=24, load_mode='mvcnn_rec', random_start_view=False):
        """
        Arguments: 
        - total_views: total views per shape; for shapenet: 24
        - num_views: number of view images per shape. Must be 24, 12, 8, 6, 4, 3, 2 or 1
        - load_mode: mvcnn_rec - load images, labels and voxels; mvcnn - load only images and labels
        """
        super().__init__()
        assert split in ['train', 'test', 'val', 'overfit']
        self.items = []
        self.classes = {}
        self.items_by_class = {}
        
        self.total_views = total_views
        self.num_views = num_views
        self.stride = self.total_views // self.num_views
        self.load_mode = load_mode
        self.random_start_view = random_start_view
        with open(cfg.DATASETS.SHAPENET.TAXONOMY_FILE_PATH, encoding='utf-8') as file:
            content = file.read()

        # print(cfg.DATASETS.SHAPENET.TAXONOMY_FILE_PATH)
        dataset_taxonomy = json.loads(content)
        #print(dataset_taxonomy)

        for taxonomy in dataset_taxonomy:
            self.classes[taxonomy['taxonomy_id']] = taxonomy['taxonomy_name']
            print('"%s": "%s"' % (taxonomy['taxonomy_name'], taxonomy['taxonomy_id']))
            self.items_by_class[taxonomy['taxonomy_name']] = []
            for item in taxonomy.get(split, []):
                self.items.append("%s/%s" % (taxonomy['taxonomy_id'], item))
                self.items_by_class[taxonomy['taxonomy_name']].append("%s/%s" % (taxonomy['taxonomy_id'], item))

    def __getitem__(self, index):
        item = self.items[index]
        # print(f"item: {item}")
        item_class = item.split('/')[0]
        start_view_idx = 0
        if self.random_start_view:
            start_view_idx = np.random.randint(0, self.stride)
        images = ShapeNetMultiview.get_images(item, self.num_views, self.total_views, start_view_idx)
        if (self.load_mode == 'mvcnn_rec'):
            voxel = ShapeNetMultiview.get_voxel(item)
            return {
                "name": item,
                "item": images,
                "voxel": voxel, # TODO check if should be new_axis here.
                "label": ShapeNetMultiview.class_ids[self.classes[item_class]]
            }
        else:
            return {
                "name": item,
                "item": images,
                "label": ShapeNetMultiview.class_ids[self.classes[item_class]]
            }

    def __len__(self):
        return len(self.items)
    
    def get_item_by_class(self, index, class_name):
        item = self.items_by_class[class_name][index]
        # print(f"item: {item}")
        item_class = item.split('/')[0]
        start_view_idx = 0
        if self.random_start_view:
            start_view_idx = np.random.randint(0, self.stride)
        images = ShapeNetMultiview.get_images(item, self.num_views, self.total_views, start_view_idx)
        if (self.load_mode == 'mvcnn_rec'):
            voxel = ShapeNetMultiview.get_voxel(item)
            return {
                "name": item,
                "item": images,
                "voxel": voxel, # TODO check if should be new_axis here.
                "label": ShapeNetMultiview.class_ids[self.classes[item_class]]
            }
        else:
            return {
                "name": item,
                "item": images,
                "label": ShapeNetMultiview.class_ids[self.classes[item_class]]
            }

    @staticmethod
    def get_voxel(item):
        voxel_path = cfg.DATASETS.SHAPENET.VOXEL_PATH % (item.split('/')[0], item.split('/')[1])
        with open(voxel_path, "rb") as voxel_file:
            voxels = read_as_3d_array(voxel_file).astype(np.float32)
        return voxels

    @staticmethod
    def get_images(item, num_views=24, total_views=24, start_idx=0):
        preprocessed_images = []
        # TODO: different transform on Train and Test mode
        preprocess = transforms.Compose([
            transforms.Resize(224), # 224 for Resnet
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Correct normalize. # output[i] = (input[i] - mean) / std
        ])
        # TODO: Custome num views 24, 12, 8, 6, 4, 3, 2, 1
        stride = total_views // num_views
        for i in range(start_idx, total_views, stride):
            img_path = cfg.DATASETS.SHAPENET.RENDERING_PATH % (item.split('/')[0], item.split('/')[1], i)
            input_image = Image.open(img_path).convert('RGB')
            input_tensor = preprocess(input_image)
            preprocessed_images.append(input_tensor)
        return torch.stack(preprocessed_images) # return a tensor


    @staticmethod
    def move_batch_to_device(batch, device, load_mode='mvcnn_rec'):
        batch["item"] = batch['item'].to(device, torch.float32)
        batch['label'] = batch['label'].to(device, torch.long)
        if load_mode == 'mvcnn_rec':
            batch['voxel'] = batch['voxel'].to(device, torch.float32)

