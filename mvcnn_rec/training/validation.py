from pathlib import Path

import torch

from mvcnn_rec.model.mvcnn import MVCNN
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