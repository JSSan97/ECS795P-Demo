import argparse
import torch
import torch.nn as nn
import time
import numpy as np
from utils import get_model, get_dataset, plot_loss, plot_accuracy, get_optimizer
from logger import setup_custom_logger

def train_loop(dataloader, model, criterion, optimizer, device, logger):
    size = len(dataloader.dataset)
    loss_ep = 0
    correct = 0

    for batch, (data, targets) in enumerate(dataloader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        predictions = model(data)
        loss = criterion(predictions, targets)

        ## Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_ep += loss.item()

        # Add correct
        correct += (predictions.argmax(1) == targets).type(torch.float).sum().item()

    avg_loss = loss_ep / len(dataloader)
    accuracy = (correct / size) * 100

    logger.info("Train Accuracy: {:>0.1f}%, Avg Loss: {:.3f}".format(accuracy, avg_loss))
    return avg_loss, accuracy

def test_loop(dataloader, model, criterion, device, logger):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for data, targets in dataloader:
            data = data.to(device=device)
            targets = targets.to(device=device)

            predictions = model(data)
            test_loss += criterion(predictions, targets).item()
            correct += (predictions.argmax(1) == targets).type(torch.float).sum().item()

    avg_loss = (test_loss / num_batches)
    accuracy = (correct / size) * 100
    logger.info("Test Accuracy: {:>0.1f}%, Avg Loss: {:.3f}".format(accuracy, avg_loss))
    return avg_loss, accuracy

def main():
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'CIFAR10', 'MNISTFashion'], help='Training/Test Dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Training Epochs')
    parser.add_argument('--model_name', type=str, default='VGG13', choices=['VGG16', 'ResNet101', 'ResNet101SE', 'ResidualAttention56'], help='Name of architecture')
    opt = parser.parse_args()

    # training parameters
    batch_size = opt.batch_size
    epochs = opt.epochs
    learning_rate = 1e-4

    # Get dataset
    dataset = get_dataset(opt.dataset, opt.model_name, batch_size)
    train_loader = dataset.get_train_loader()
    test_loader = dataset.get_test_loader()
    results_path = dataset.get_results_model_dir()

    # initialise the device for training, if gpu is available, device = 'cuda', else: device = 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get model from model name
    model = get_model(opt.model_name, device, dataset)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, learning_rate)

    # Keep record of loss, accuracy
    history = {}
    history['train_avg_loss'] = []
    history['train_accuracy'] = []
    history['test_avg_loss'] = []
    history['test_accuracy'] = []
    history['time'] = []

    logger_filename = 'LOG_{}_{}_{}.txt'.format(opt.model_name, opt.dataset_name, opt.epoch)
    logger = setup_custom_logger('logger', logger_filename)
    logger.info("Training Model: {}, Dataset: {}, Epochs {}".format(opt.model_name, opt.dataset, epochs))

    # Training
    start_time = time.time()
    for epoch in range(epochs):
        logger.info("---- Epoch {} ----".format(epoch + 1))

        train_avg_loss, train_accuracy = train_loop(train_loader, model, criterion, optimizer, device, logger)
        history['train_avg_loss'].append(train_avg_loss)
        history['train_accuracy'].append(train_accuracy)
        test_avg_loss, test_accuracy = test_loop(test_loader, model, criterion, device, logger)
        history['test_avg_loss'].append(test_avg_loss)
        history['test_accuracy'].append(test_accuracy)
        current_time = time.time() - start_time
        history['time'].append(current_time)

        logger.info("Current Training Time: {}".format(current_time))

    end_time = time.time()
    total_train_time = end_time - start_time
    logger.info("Total Training Time: {}".format(total_train_time))

    # Save training/test history
    history_filename = "training_history_{}.npy".format(epoch + 1)
    np.save('{}/{}'.format(results_path, history_filename), history)

    # Save Model
    model_filename = "model_{}_epoch{}.pth".format(opt.model_name, epoch + 1)
    torch.save(model.state_dict(), "{}/{}".format(results_path, model_filename))

    # Show/Save Training/Testing Process
    loss_filename = "{}_loss_over_epoch_{}".format(opt.model_name, epochs)
    plot_loss(history, show=False, save=True, path='{}/{}'.format(results_path, loss_filename))

    accuracy_filename = "{}_accuracy_over_epoch_{}".format(opt.model_name, epochs)
    plot_accuracy(history, show=False, save=True, path='{}/{}'.format(results_path, accuracy_filename))

if __name__ == '__main__':
    main()

