import numpy as np
from mvcnn_rec.data.shapenet import ShapeNetMultiview
from mvcnn_rec.inference.infer_mvcnn_rec import InferenceHandlerMVCNNReconstruction
from mvcnn_rec.training.validation import compute_iou_2_voxels
from mvcnn_rec.util.marching_cubes import marching_cubes
from mvcnn_rec.util.export import export_mesh_to_obj

class MVCNN_Rec_Visualizer():
    def __init__(self, ckpt, num_views, voxel_thresh=0.3):
        self.val_dataset = ShapeNetMultiview('val', total_views=24, num_views=num_views, 
                                load_mode='mvcnn_rec', # Change to mvcnn to get only images and labels
                                random_start_view=False)

        self.test_dataset = ShapeNetMultiview('test', total_views=24, num_views=num_views, 
                                        load_mode='mvcnn_rec', # Change to mvcnn to get only images and labels
                                        random_start_view=False)
        
        self.train_dataset = ShapeNetMultiview('train', total_views=24, num_views=12, 
                                  load_mode='mvcnn_rec', # Change to mvcnn to get only images and labels
                                  random_start_view=False)
        
        self.datasets = {
            'train': self.train_dataset, 
            'val': self.val_dataset,
            'test': self.test_dataset
        }
        
        self.datasets_length = {
            'train': len(self.train_dataset), 
            'val': len(self.val_dataset),
            'test': len(self.test_dataset)
        }
        
        self.model_name = ckpt.split('/')[-1].split('.')[0]
        self.inferer = InferenceHandlerMVCNNReconstruction(ckpt)
        self.voxel_thresh = voxel_thresh
        self.num_views =num_views

    def get_shape(self, shape_id, data_name='test', save_voxels=False, folder='visualization'):
        assert(shape_id < self.datasets_length[data_name])
        sample = self.datasets[data_name][shape_id]
        
        print(f'{data_name} dataset - Id {shape_id}')
        print(f'Name: {sample["name"]}')
        print(f'Images Dimensions: {sample["item"].shape}')
        print(f'Voxel Dimensions: {sample["voxel"].shape}')
        print(f'Label: {ShapeNetMultiview.id_class_mapping[sample["label"]]}')

        if save_voxels:
            fname =  f'{folder}/{ShapeNetMultiview.id_class_mapping[sample["label"]]}_{shape_id}_groundtruth.obj'
            print(f'Saving to {fname}.')
            MVCNN_Rec_Visualizer.voxels_to_obj_file(sample["voxel"].astype(int), fname)

        return sample

    def get_shape_id_class(self, idx, class_name, data_name='test', save_voxels=False):
        shape_id = self.datasets[data_name].index_class_to_global_idx(idx, class_name)
        return shape_id
    
    def get_predict_shape_id(self, shape_id, data_name='test', voxel_thresh=0.5, save_voxels=True, folder='visualization'):
        sample = self.get_shape(shape_id, data_name)
        class_name, predited_voxel = self.inferer.infer_single(sample['item'], voxel_thresh)
        iou = compute_iou_2_voxels(predited_voxel, sample["voxel"])

        # Save sample['voxel'] and predicted_voxel 
        # Currently np 3d array as occupancy grid 32x32x32
        if save_voxels:
            fname =  f'{folder}/{ShapeNetMultiview.id_class_mapping[sample["label"]]}_{shape_id}_{self.model_name}_{self.num_views}_views_{iou:.2f}_IoU.obj'
            print(f'Saving to {fname}.')
            MVCNN_Rec_Visualizer.voxels_to_obj_file(predited_voxel.astype(int), fname)
        return sample, class_name, predited_voxel, iou

    @staticmethod
    def voxels_to_obj_file(voxels, fname):
        sdf = np.where(voxels > 0, -0.5, 0.5).astype(np.float64)
        vertices, faces = marching_cubes(sdf)
        export_mesh_to_obj(fname, vertices, faces)
    
    
    def random_shape(self, data_name='test'):
        shape_id = np.random.randint(self.datasets_length[data_name])
        return self.get_predict_shape_id(shape_id, data_name)

    def random_shape_id(self, data_name='test'):
        shape_id = np.random.randint(self.datasets_length[data_name])
        return shape_id

        