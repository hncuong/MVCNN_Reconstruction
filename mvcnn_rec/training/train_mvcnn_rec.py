from pathlib import Path

import torch

from mvcnn_rec.model.mvcnn_rec import MVCNNReconstruction
from mvcnn_rec.data.shapenet import ShapeNetMultiview

def train(model, trainloader, valloader, device, config):
    loss_class_criterion = torch.nn.CrossEntropyLoss()
    loss_class_criterion = loss_class_criterion.to(device)

    # TODO Reconstruction loss for voxels. Change to BCEWithLogitsLoss if output is logits.
    # loss_voxels = torch.nn.BCEWithLogitsLoss() 
    loss_voxels_criterion = torch.nn.BCELoss()
    loss_voxels_criterion = loss_voxels_criterion.to(device)

    # TODO Weight of losses
    loss_class_weight = config.get('loss_class_weight', 0.5)
    loss_voxel_weight = config.get('loss_voxel_weight', 0.5)
    voxel_thresh = config.get('voxel_thresh', 0.3)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # lr scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # set model to train, important if your network has e.g. dropout or batchnorm layers
    model.train()

    best_accuracy = 0.
    best_val_loss = 1000.

    # keep track of running average of train loss for printing
    train_loss_running = 0.

    for epoch in range(config['max_epochs']):
        for i, batch in enumerate(trainloader):
            # move batch to device
            ShapeNetMultiview.move_batch_to_device(batch, device)
            input_data, target_labels, target_voxels = batch['item'], batch['label'], batch['voxel']
            # TODO Reshape inpute_data to [N_images, B, 3, H, W] (H, W = 222)
            # N,B,C,H,W = input_data.size()
            # input_data = input_data.view(B, N, C, H, W)
            input_data = torch.swapaxes(input_data, 0, 1) # TODO Need to fix

            optimizer.zero_grad()

            pred_class, pred_voxels = model(input_data)

            # TODO loss voxels
            loss_class = loss_class_criterion(pred_class, target_labels)
            loss_voxel = loss_voxels_criterion(pred_voxels, target_voxels)
            loss = loss_class_weight * loss_class + loss_voxel_weight * loss_voxel

            loss.backward()

            optimizer.step()

            # loss logging
            train_loss_running += loss.item()
            iteration = epoch * len(trainloader) + i

            if iteration % config['print_every_n'] == (config['print_every_n'] - 1):
                print(f'[{epoch:03d}/{i:05d}] train_loss: {train_loss_running / config["print_every_n"]:.3f}')
                train_loss_running = 0.

            # validation evaluation and logging
            if iteration % config['validate_every_n'] == (config['validate_every_n'] - 1):

                # set model to eval, important if your network has e.g. dropout or batchnorm layers
                model.eval()

                loss_total_val = 0
                total, correct = 0, 0
                ious = []

                # forward pass and evaluation for entire validation set
                for batch_val in valloader:
                    ShapeNetMultiview.move_batch_to_device(batch_val, device)
                    input_data, target_labels, target_voxels = batch_val['images'], batch_val['label'], batch_val['voxel']
                    input_data = torch.swapaxes(input_data, 0, 1) # TODO Need to fix

                    with torch.no_grad():
                        # prediction = model(input_data)
                        pred_class, pred_voxels = model(input_data)
                    
                    loss_total_val += loss_class_weight * loss_class_criterion(pred_class, target_labels).item() + \
                            loss_voxel_weight * loss_voxels_criterion(pred_voxels, target_voxels).item()
                          
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
                    

                accuracy = 100 * correct / total
                # TODO: mean IoU
                mean_IoU = torch.mean(torch.stack(ious)).item()
                val_loss = loss_total_val / len(valloader)

                print(f'[{epoch:03d}/{i:05d}] val_loss: {val_loss:.3f}, val_accuracy: {accuracy:.3f}%, val_IoU: {mean_IoU}.')

                # TODO change condition if needed
                if val_loss < best_val_loss:
                    torch.save(model.state_dict(), f'mvcnn_rec/runs/{config["experiment_name"]}/model_best.ckpt')
                    best_val_loss = val_loss

                # set model back to train
                model.train()

def main(config):
    """
    Function for training MVCNN on ShapeNet
    :param config: configuration for training - has the following keys
                   'experiment_name': name of the experiment, checkpoint will be saved to folder "exercise_2/runs/<experiment_name>"
                   'device': device on which model is trained, e.g. 'cpu' or 'cuda:0'
                   'batch_size': batch size for training and validation dataloaders
                   'resume_ckpt': None if training from scratch, otherwise path to checkpoint (saved weights)
                   'learning_rate': learning rate for optimizer
                   'max_epochs': total number of epochs after which training should stop
                   'print_every_n': print train loss every n iterations
                   'validate_every_n': print validation loss and validation accuracy every n iterations
                   'is_overfit': if the training is done on a small subset of data specified in exercise_2/split/overfit.txt,
                                 train and validation done on the same set, so error close to 0 means a good overfit. Useful for debugging.
    """

    # Declare device
    device = torch.device('cpu')
    if torch.cuda.is_available() and config['device'].startswith('cuda'):
        device = torch.device(config['device'])
        print('Using device:', config['device'])
    else:
        print('Using CPU')

    # Create Dataloaders
    train_dataset = ShapeNetMultiview('train' if not config['is_overfit'] else 'overfit', total_views=24, 
                                    num_views=config.get('num_views', 12), load_mode='mvcnn_rec', 
                                    random_start_view=config.get('random_start_view', False))
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,   # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config['batch_size'],   # The size of batches is defined here
        shuffle=True,    # Shuffling the order of samples is useful during training to prevent that the network learns to depend on the order of the input data
        num_workers=4,   # Data is usually loaded in parallel by num_workers
        pin_memory=True  # This is an implementation detail to speed up data uploading to the GPU
    )

    val_dataset = ShapeNetMultiview('val' if not config['is_overfit'] else 'overfit', total_views=24, 
                                    num_views=config.get('num_views', 12), load_mode='mvcnn_rec',
                                    random_start_view=False)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,     # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config['batch_size'],   # The size of batches is defined here
        shuffle=False,   # During validation, shuffling is not necessary anymore
        num_workers=4,   # Data is usually loaded in parallel by num_workers
        pin_memory=True  # This is an implementation detail to speed up data uploading to the GPU
    )

    # Instantiate model
    # TODO: Update model later.
    model = MVCNNReconstruction(ShapeNetMultiview.num_classes)

    # Load model if resuming from checkpoint
    if config['resume_ckpt'] is not None:
        model.load_state_dict(torch.load(config['resume_ckpt'], map_location='cpu'))

    # Move model to specified device
    model.to(device)

    # Create folder for saving checkpoints
    Path(f'mvcnn_rec/runs/{config["experiment_name"]}').mkdir(exist_ok=True, parents=True)

    # Start training
    train(model, train_dataloader, val_dataloader, device, config)
