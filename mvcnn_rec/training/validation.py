from pathlib import Path

import torch
import numpy as np

from mvcnn_rec.model.mvcnn import MVCNN
from mvcnn_rec.model.mvcnn_rec import MVCNNReconstruction
from mvcnn_rec.data.shapenet import ShapeNetMultiview
from datetime import datetime


def validation_mvcnn(model, val_dataset, config):
    """ Validation a model MVCNN on a given dataset with given config
    Arguments:
    - model: MVCNN model
    - val_dataset: dataset to evaluate
    - config: get batch size and shuffle.
    """
    device = torch.device('cpu')
    if torch.cuda.is_available() and config['device'].startswith('cuda'):
        device = torch.device(config['device'])
        print('Using device:', config['device'])
    else:
        print('Using CPU')
    # set model to eval, important if your network has e.g. dropout or batchnorm layers
    model.to(device)
    model.eval()
    
    valloader = torch.utils.data.DataLoader(
        val_dataset,     # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config.get('batch_size', 8),   # The size of batches is defined here
        shuffle=config.get('shuffle', False),   # During validation, shuffling is not necessary anymore
        num_workers=4,   # Data is usually loaded in parallel by num_workers
        pin_memory=True  # This is an implementation detail to speed up data uploading to the GPU
    )

    loss_criterion = torch.nn.CrossEntropyLoss()
    loss_criterion = loss_criterion.to(device)


    loss_total_val = 0
    total, correct = 0, 0
    # forward pass and evaluation for entire validation set
    for idx, batch_val in enumerate(valloader):
        ShapeNetMultiview.move_batch_to_device(batch_val, device, load_mode='mvcnn')
        input_data, target_labels = batch_val['item'], batch_val['label']
        input_data = torch.swapaxes(input_data, 0, 1) # Fixed

        with torch.no_grad():
            prediction = model(input_data)

        predicted_label = torch.argmax(prediction, dim=1)
        total += predicted_label.shape[0]
        correct += (predicted_label == target_labels).sum().item()
        loss = loss_criterion(prediction, target_labels)
        if idx % config.get('print_every_n', 50) == 0:
            print(f"Batch id {idx} - Loss: {loss.item()}")
            # print(target_labels)
        loss_total_val += loss.item()

    accuracy = 100 * correct / total
    print(f'Num batch val {idx}')
    print(f'{datetime.now()} val_loss: {loss_total_val / len(valloader):.3f}, val_accuracy: {accuracy:.3f}%')


def validation_mvcnn_reconstruction(model, val_dataset, config):
    """ Validation a model MVCNN Reconstruction on a given dataset with given config
    Arguments:
    - model: MVCNN model
    - val_dataset: dataset to evaluate
    - config: get batch size and shuffle.
    """
    device = torch.device('cpu')
    if torch.cuda.is_available() and config['device'].startswith('cuda'):
        device = torch.device(config['device'])
        print('Using device:', config['device'])
    else:
        print('Using CPU')
    # set model to eval, important if your network has e.g. dropout or batchnorm layers
    model.to(device)
    model.eval()
    
    valloader = torch.utils.data.DataLoader(
        val_dataset,     # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config.get('batch_size', 8),   # The size of batches is defined here
        shuffle=config.get('shuffle', False),   # During validation, shuffling is not necessary anymore
        num_workers=4,   # Data is usually loaded in parallel by num_workers
        pin_memory=True  # This is an implementation detail to speed up data uploading to the GPU
    )

    voxel_thresh = config.get('voxel_thresh', 0.3)

    total, correct = 0, 0
    ious = []

    # forward pass and evaluation for entire validation set
    for idx, batch_val in enumerate(valloader):
        ShapeNetMultiview.move_batch_to_device(batch_val, device)
        input_data, target_labels, target_voxels = batch_val['item'], batch_val['label'], batch_val['voxel']
        input_data = torch.swapaxes(input_data, 0, 1)

        with torch.no_grad():
            # prediction = model(input_data)
            pred_voxels, pred_class = model(input_data)
            
        predicted_label = torch.argmax(pred_class, dim=1)
        # TODO binarize voxels prediction. Get binarize threshold from config. DONE
        # Assumed its is output of Sigmoid
        predicted_voxel = torch.where(pred_voxels > voxel_thresh, 1, 0)

        # Class eval
        total += predicted_label.shape[0]
        correct += (predicted_label == target_labels).sum().item()
        # Voxel eval
        I = torch.sum(torch.logical_and(predicted_voxel == 1, target_voxels == 1))
        U = torch.sum(torch.logical_or(predicted_voxel == 1, target_voxels == 1))
        if U == 0:
            iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
        else:
            iou = I / float(U)
        ious.append(iou)

        if idx % config.get('print_every_n', 50) == 0:
            accuracy = 100 * correct / total
            mean_IoU = torch.mean(torch.stack(ious)).item()
            print(f'{datetime.now()} Batch id {idx}: val_accuracy: {accuracy:.3f}%, val_IoU: {mean_IoU}.')

    accuracy = 100 * correct / total
    mean_IoU = torch.mean(torch.stack(ious)).item()

    print(f'{datetime.now()}: val_accuracy: {accuracy:.3f}%, val_IoU: {mean_IoU}.')

def validation_mvcnn_rec(model, valloader, config, device):
    # set model to eval, important if your network has e.g. dropout or batchnorm layers
    print(f"{datetime.now()} Start validating!")
    model.eval()
    voxel_thresh = config.get('voxel_thresh', 0.3)

    total, correct = 0, 0
    ious = []

    # forward pass and evaluation for entire validation set
    for batch_id, batch_val in enumerate(valloader):
        ShapeNetMultiview.move_batch_to_device(batch_val, device)
        input_data, target_labels, target_voxels = batch_val['item'], batch_val['label'], batch_val['voxel']
        input_data = torch.swapaxes(input_data, 0, 1) # TODO Need to fix
        # target_voxels = torch.squeeze(target_voxels, dim = 1)

        with torch.no_grad():
            # prediction = model(input_data)
            pred_voxels, pred_class = model(input_data)
                
        predicted_label = torch.argmax(pred_class, dim=1)
        # TODO binarize voxels prediction. Get binarize threshold from config. DONE
        # Assumed its is output of Sigmoid
        predicted_voxel = torch.where(pred_voxels > voxel_thresh, 1, 0)

        # Class eval
        total += predicted_label.shape[0]
        correct += (predicted_label == target_labels).sum().item()
        # TODO Voxel eval
        I = torch.sum(torch.logical_and(predicted_voxel == 1, target_voxels == 1))
        U = torch.sum(torch.logical_or(predicted_voxel == 1, target_voxels == 1))
        if U == 0:
            iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
        else:
            iou = I / float(U)
        ious.append(iou)

        if batch_id % config.get('print_every_n', 50) == config.get('print_every_n', 50) - 1:
            accuracy = 100 * correct / total
            mean_IoU = torch.mean(torch.stack(ious)).item()
            print(f'{datetime.now()} Batch id {batch_id}: val_accuracy: {accuracy:.3f}%, val_IoU: {mean_IoU}.')
        

    accuracy = 100 * correct / total
    # TODO: mean IoU
    mean_IoU = torch.mean(torch.stack(ious)).item()

    print(f'{datetime.now()}: Test_accuracy: {accuracy:.3f}%, Test_IoU: {mean_IoU}.')

def validation_mvcnn_rec_by_class(model, valloader, config, device):
    # set model to eval, important if your network has e.g. dropout or batchnorm layers
    print(f"{datetime.now()} Start validating!")
    model.eval()
    voxel_thresh = config.get('voxel_thresh', 0.3)

    total = [0] * ShapeNetMultiview.num_classes
    correct = [0] * ShapeNetMultiview.num_classes
    total_gt = [0] * ShapeNetMultiview.num_classes
    ious = []
    ious_class = [0.] * ShapeNetMultiview.num_classes

    # forward pass and evaluation for entire validation set
    for batch_id, batch_val in enumerate(valloader):
        ShapeNetMultiview.move_batch_to_device(batch_val, device)
        input_data, target_labels, target_voxels = batch_val['item'], batch_val['label'], batch_val['voxel']
        input_data = torch.swapaxes(input_data, 0, 1) # TODO Need to fix
        # target_voxels = torch.squeeze(target_voxels, dim = 1)

        with torch.no_grad():
            # prediction = model(input_data)
            pred_voxels, pred_class = model(input_data)
                
        predicted_label = torch.argmax(pred_class, dim=1)
        # TODO binarize voxels prediction. Get binarize threshold from config. DONE
        # Assumed its is output of Sigmoid
        predicted_voxel = torch.where(pred_voxels > voxel_thresh, 1, 0)

        # Class eval
        for i in range(ShapeNetMultiview.num_classes):
            n_pred_i = torch.sum(predicted_label == i).item()
            total[i] += n_pred_i
            if n_pred_i > 0:
                correct[i] += torch.sum(torch.logical_and(predicted_label == i, predicted_label == target_labels)).item()

            n_gt_i = torch.sum(target_labels == i).item()
            total_gt[i] += n_gt_i
            if n_gt_i > 0:
                target_voxels_i = target_voxels[target_labels == i]
                predicted_voxel_i = predicted_voxel[target_labels == i]
                I = torch.sum(torch.logical_and(predicted_voxel_i == 1, target_voxels_i == 1))
                U = torch.sum(torch.logical_or(predicted_voxel_i == 1, target_voxels_i == 1))
            if U == 0:
                iou = n_gt_i  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U) * n_gt_i
            # ious.append(iou)
            ious_class[i] += iou

        # TODO Voxel eval
        I = torch.sum(torch.logical_and(predicted_voxel == 1, target_voxels == 1))
        U = torch.sum(torch.logical_or(predicted_voxel == 1, target_voxels == 1))
        if U == 0:
            iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
        else:
            iou = I / float(U)
        ious.append(iou)

        if batch_id % config.get('print_every_n', 50) == config.get('print_every_n', 50) - 1:
            print('\nAccuracy by class:')
            for i in range(ShapeNetMultiview.num_classes):
                if total[i] > 0:
                    accuracy = 100 * correct[i] / total[i]
                    print(f'{ShapeNetMultiview.id_class_mapping[i]}: {accuracy:.3f}%.')

            print('\nMean IoU by class:')
            for i in range(ShapeNetMultiview.num_classes):
                if total_gt[i] > 0:
                    mean_IoU = ious_class[i] / total_gt[i]
                    print(f'{ShapeNetMultiview.id_class_mapping[i]}: {mean_IoU:.3f}.')

            accuracy = 100 * sum(correct) / sum(total)
            mean_IoU = torch.mean(torch.stack(ious)).item()
            print(f'{datetime.now()} Batch id {batch_id}: val_accuracy: {accuracy:.3f}%, val_IoU: {mean_IoU}.')
        
    print('\nAccuracy by class:')
    for i in range(ShapeNetMultiview.num_classes):
        if total[i] > 0:
            accuracy = 100 * correct[i] / total[i]
            print(f'{ShapeNetMultiview.id_class_mapping[i]}: {accuracy:.1f}%.')
        
    print('\nIoU by class:')
    for i in range(ShapeNetMultiview.num_classes):
        if total_gt[i] > 0:
            mean_IoU = ious_class[i] / total_gt[i]
            print(f'{ShapeNetMultiview.id_class_mapping[i]}: {mean_IoU:.2f}')

    accuracy = 100 * sum(correct) / sum(total)
    # accuracy = 100 * correct / total
    # TODO: mean IoU
    mean_IoU = torch.mean(torch.stack(ious)).item()
    print(f'{datetime.now()}: Test_accuracy: {accuracy:.1f}%, Test_IoU: {mean_IoU}.') 

def compute_iou_2_voxels(predicted_voxel, target_voxels):
    I = np.sum(np.logical_and(predicted_voxel == 1, target_voxels == 1))
    U = np.sum(np.logical_or(predicted_voxel == 1, target_voxels == 1))
    if U == 0:
        iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
    else:
        iou = I / float(U)
    return iou


def main(config):
    # Declare device
    device = torch.device('cpu')
    if torch.cuda.is_available() and config['device'].startswith('cuda'):
        device = torch.device(config['device'])
        print('Using device:', config['device'])
    else:
        print('Using CPU')

    val_dataset = ShapeNetMultiview(config.get('dataset', 'test') if not config['is_overfit'] else 'overfit', total_views=24, 
                                    num_views=config.get('num_views', 12), load_mode='mvcnn_rec',
                                    random_start_view=False)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,     # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config['batch_size'],   # The size of batches is defined here
        shuffle=config.get('shuffle', False),   # During validation, shuffling is not necessary anymore
        num_workers=config.get('num_workers', 4),   # Data is usually loaded in parallel by num_workers
        pin_memory=True  # This is an implementation detail to speed up data uploading to the GPU
    )

    model = MVCNNReconstruction(ShapeNetMultiview.num_classes)

    # Load model if resuming from checkpoint
    if config['resume_ckpt'] is not None:
        model.load_state_dict(torch.load(config['resume_ckpt'], map_location='cpu'))

    # Move model to specified device
    model.to(device)
    if config.get('validation_mode', 'all') == 'all':
        validation_mvcnn_rec(model, val_dataloader, config, device)
    elif config.get('validation_mode', 'all') == 'class':
        validation_mvcnn_rec_by_class(model, val_dataloader, config, device)

