from pathlib import Path

import torch

# TODO update model name
from mvcnn_rec.model.mvcnn import MVCNN
from mvcnn_rec.data.shapenet import ShapeNetMultiview

def train(model, trainloader, valloader, device, config):
    loss_criterion = torch.nn.CrossEntropyLoss()
    loss_criterion = loss_criterion.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # lr scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # set model to train, important if your network has e.g. dropout or batchnorm layers
    model.train()

    best_accuracy = 0.

    # keep track of running average of train loss for printing
    train_loss_running = 0.

    for epoch in range(config['max_epochs']):
        for i, batch in enumerate(trainloader):
            # move batch to device
            ShapeNetMultiview.move_batch_to_device(batch, device)
            input_data, target_labels = batch['images'], batch['label']

            optimizer.zero_grad()

            prediction = model(input_data)

            loss = loss_criterion(prediction, target_labels)

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
                # forward pass and evaluation for entire validation set
                for batch_val in valloader:
                    ShapeNetMultiview.move_batch_to_device(batch_val, device)
                    input_data, target_labels = batch_val['images'], batch_val['label']

                    with torch.no_grad():
                        prediction = model(input_data)

                    predicted_label = torch.argmax(prediction, dim=1)

                    total += predicted_label.shape[0]
                    correct += (predicted_label == target_labels).sum().item()

                    loss_total_val += loss_criterion(prediction, target_labels).item()

                accuracy = 100 * correct / total

                print(f'[{epoch:03d}/{i:05d}] val_loss: {loss_total_val / len(valloader):.3f}, val_accuracy: {accuracy:.3f}%')

                if accuracy > best_accuracy:
                    torch.save(model.state_dict(), f'mvcnn_rec/runs/{config["experiment_name"]}/model_best.ckpt')
                    best_accuracy = accuracy

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
    train_dataset = ShapeNetMultiview('train' if not config['is_overfit'] else 'overfit')
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,   # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config['batch_size'],   # The size of batches is defined here
        shuffle=True,    # Shuffling the order of samples is useful during training to prevent that the network learns to depend on the order of the input data
        num_workers=4,   # Data is usually loaded in parallel by num_workers
        pin_memory=True  # This is an implementation detail to speed up data uploading to the GPU
    )

    val_dataset = ShapeNetMultiview('val' if not config['is_overfit'] else 'overfit')
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,     # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config['batch_size'],   # The size of batches is defined here
        shuffle=False,   # During validation, shuffling is not necessary anymore
        num_workers=4,   # Data is usually loaded in parallel by num_workers
        pin_memory=True  # This is an implementation detail to speed up data uploading to the GPU
    )

    # Instantiate model
    # TODO: Update model later.
    model = MVCNN(ShapeNetMultiview.num_classes)

    # Load model if resuming from checkpoint
    if config['resume_ckpt'] is not None:
        model.load_state_dict(torch.load(config['resume_ckpt'], map_location='cpu'))

    # Move model to specified device
    model.to(device)

    # Create folder for saving checkpoints
    Path(f'mvcnn_rec/runs/{config["experiment_name"]}').mkdir(exist_ok=True, parents=True)

    # Start training
    train(model, train_dataloader, val_dataloader, device, config)
