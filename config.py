from easydict import EasyDict as edict

__C = edict()
cfg = __C
#
# Dataset Config
#
__C.DATASETS = edict()
__C.DATASETS.SHAPENET = edict()

__C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH    = './datasets/ShapeNet.json'
__C.DATASETS.SHAPENET.RENDERING_PATH = './datasets/ShapeNetRendering/%s/%s/rendering/%02d.png'
__C.DATASETS.SHAPENET.VOXEL_PATH = './datasets/ShapeNetVox32/%s/%s/model.binvox'