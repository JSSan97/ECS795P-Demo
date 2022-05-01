import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

from models.residualattention import ResidualAttentionModel_56, ResidualAttentionModel_92
from models.vgg import VGG16
from models.resnet import ResNet101, ResNet101SE, ResNet101CBAM
from datasets import MNISTDigits, CIFAR10, FashionMNIST


def load_image_from_tensor(image, save_path=None, title=None):
    image = image / 2 + 0.5
    np_image = image.numpy()
    plt.axis('off')

    if title:
        plt.title(title)

    plt.imshow(np.transpose(np_image, (1, 2, 0)))

    if save_path:
        # print("Images saved in: {}".format(save_path))
        plt.savefig(save_path)

    plt.show()
    plt.pause(0.1)

    plt.close()


def get_model(model_name, device, dataset):
    input_channels = 3
    if dataset.get_dataset_name() == "MNIST" or dataset.get_dataset_name() == "MNISTFashion":
        input_channels = 1

    models = {
        "VGG16": VGG16(dataset.get_number_of_classes(), input_channels),
        "ResNet101": ResNet101(dataset.get_number_of_classes(), input_channels),
        "ResNet101SE": ResNet101SE(dataset.get_number_of_classes(), input_channels),
        "ResNet101CBAM": ResNet101CBAM(dataset.get_number_of_classes(), input_channels),
        'ResidualAttention56': ResidualAttentionModel_56(dataset.get_number_of_classes(), input_channels),
        'ResidualAttention92': ResidualAttentionModel_92(dataset.get_number_of_classes(), input_channels), ## Too many parameters, run out of memory on google collab...
    }

    return models.get(model_name).to(device=device)

def get_optimizer(net, learning_rate):
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    return optimizer

def get_dataset(dataset_name, model_name, batch_size):
    datasets = {
        "MNIST": MNISTDigits(batch_size, model_name),
        "CIFAR10": CIFAR10(batch_size, model_name),
        "MNISTFashion": FashionMNIST(batch_size, model_name),
    }

    return datasets.get(dataset_name)

def plot_loss(history, show=False, save=False, path='train_loss.png'):
    x = range(len(history['train_avg_loss']))
    plt.plot(x, history['train_avg_loss'], label='Training Loss')
    plt.plot(x, history['test_avg_loss'], label='Test Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss of Test and Training Sets Over Epoch')
    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)
    if show:
        plt.show()
    else:
        plt.close()

def plot_accuracy(history, show=False, save=False, path='train_accuracy.png'):
    x = range(len(history['train_avg_loss']))

    plt.plot(x, history['train_accuracy'], label='Training Accuracy')
    plt.plot(x, history['test_accuracy'], label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Test and Training Sets Over Epoch')
    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(path)
    if show:
        plt.show()
    else:
        plt.close()