from pathlib import Path
import json
import os
import cv2
import numpy as np
import torch
from config import cfg

from mvcnn_rec.data.binvox_rw import read_as_3d_array

class ShapeNetMultiview(torch.utils.data.Dataset):
    """Class for loading multiview images, class, and voxel of object
    """
    def __init__(self, split):
        super().__init__()
        assert split in ['train', 'test', 'val']
        self.items = []
        self.classes = {}

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
            "label": self.classes[item_class]
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
        #taxonomy_folder_name = item.split('/')[0]
        #sample_name = item.split('/')[1]
        ## Get file list of rendering images
        #img_file_path = cfg.DATASETS.SHAPENET.RENDERING_PATH % (taxonomy_folder_name, sample_name, 0)
        #img_folder = os.path.dirname(img_file_path)
        #total_views = len(os.listdir(img_folder))
        #rendering_image_indexes = range(total_views)
        #rendering_images_file_path = []
        #for image_idx in rendering_image_indexes:
        #    img_file_path = cfg.DATASETS.SHAPENET.RENDERING_PATH % (taxonomy_folder_name, sample_name, image_idx)
        #    if not os.path.exists(img_file_path):
        #        continue
        #    rendering_images_file_path.append(img_file_path)

        #file_list = []
        rendering_images = []
        for i in range(0, 23):
            img_path = cfg.DATASETS.SHAPENET.RENDERING_PATH % (item.split('/')[0], item.split('/')[1], i)
            #print(f"open img_path: {img_path}")
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            #cv2.imshow('image', image)
            #cv2.waitKey(0)
            rendering_image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
            #print(rendering_image.shape)
            rendering_images.append(rendering_image)

        return np.asarray(rendering_images)


    @staticmethod
    def move_batch_to_device(batch, device):
        raise NotImplementedError
