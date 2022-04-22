import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_model, get_dataset, show_train_loss, show_train_accuracy

def main():
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
    parser.add_argument('--model_name', type=str, default='VGG13', choices=['VGG13', 'VGG16'], help='Name of architecture')
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST, CIFAR10', 'MNISTFashion'], help='Training/Test Dataset')
    parser.add_argument('--epochs', type=int, default=100, help='Training Epochs')
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
    model = get_model(opt.model_name, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Keep record of loss, accuracy
    history = {}
    history['loss_per_epoch'] = []
    history['accuracy_per_epoch'] = []

    # Training
    for epoch in range(epochs):
        loss_ep = 0

        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device=device)
            targets = targets.to(device=device)
            ## Forward Pass
            optimizer.zero_grad()
            scores = model(data)
            loss = criterion(scores, targets)
            loss.backward()
            optimizer.step()
            loss_ep += loss.item()

        loss_for_epoch = loss_ep / len(train_loader)
        history['loss_per_epoch'].append(loss_for_epoch)
        print("Loss in epoch {} - {}".format(epoch, loss_for_epoch))
        with torch.no_grad():
            num_correct = 0
            num_samples = 0
            for batch_idx, (data, targets) in enumerate(test_loader):
                data = data.to(device=device)
                targets = targets.to(device=device)
                ## Forward Pass
                scores = model(data)
                _, predictions = scores.max(1)
                num_correct += (predictions == targets).sum()
                num_samples += predictions.size(0)

            accuracy_for_epoch = float(num_correct / float(num_samples) * 100)
            history['accuracy_per_epoch'].append(accuracy_for_epoch)
            print("num_current: {} out of {}. Accuracy {:.2f}%".format(num_correct, num_samples, accuracy_for_epoch))

    # Show/Save Training/Testing Process
    loss_filename = "{}_loss_over_epoch_{}".format(opt.model_name, epochs)
    show_train_loss(history, show=False, save=True, path='{}/{}'.format(results_path, loss_filename))

    accuracy_filename = "{}_test_accuracy_over_epoch_{}".format(opt.model_name, epochs)
    show_train_accuracy(history, show=False, save=True, path='{}/{}'.format(results_path, accuracy_filename))

if __name__ == '__main__':
    main()

