import numpy as np
from mvcnn_rec.data.shapenet import ShapeNetMultiview
from mvcnn_rec.inference.infer_mvcnn_rec import InferenceHandlerMVCNNReconstruction
from mvcnn_rec.training.validation import compute_iou_2_voxels

class MVCNN_Rec_Visualizer():
    def __init__(self, ckpt, num_views, voxel_thresh=0.5):
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
        
        self.inferer = InferenceHandlerMVCNNReconstruction(ckpt)
        self.voxel_thresh = voxel_thresh

    def get_shape(self, shape_id, data_name='test'):
        assert(shape_id < self.datasets_length[data_name])
        sample = self.datasets[data_name][shape_id]
        
        print(f'{data_name} dataset - Id {shape_id}')
        print(f'Name: {sample["name"]}')
        print(f'Images Dimensions: {sample["item"].shape}')
        print(f'Voxel Dimensions: {sample["voxel"].shape}')
        print(f'Label: {ShapeNetMultiview.id_class_mapping[sample["label"]]}')
        return sample
    
    def get_predict_shape_id(self, shape_id, data_name='test', voxel_thresh=0.5):
        sample = self.get_shape(shape_id, data_name)
        class_name, predited_voxel = self.inferer.infer_single(sample['item'], voxel_thresh)
        iou = compute_iou_2_voxels(predited_voxel, sample["voxel"])
        return sample, class_name, predited_voxel, iou
    
    def random_shape(self, data_name='test'):
        shape_id = np.random.randint(self.datasets_length[data_name])
        return self.get_predict_shape_id(shape_id, data_name)

    def random_shape_id(self, data_name='test'):
        shape_id = np.random.randint(self.datasets_length[data_name])
        return shape_id

        