import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from models import VGG16, VGG13
from datasets import MNISTDigits, CIFAR10, FashionMNIST

def view_images(data_loader):
    for images, _ in data_loader:
        print('images.shape:', images.shape)
        plt.figure(figsize=(16,8))
        plt.axis('off')
        plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
        break

def get_model(model_name, device):
    models = {
        "VGG13": VGG13(),
        "VGG16": VGG16(),
    }

    return models.get(model_name).to(device=device)

def get_dataset(dataset_name, model_name, batch_size):
    datasets = {
        "MNIST": MNISTDigits(batch_size, model_name),
        "CIFAR10": CIFAR10(batch_size, model_name),
        "ImageNet": FashionMNIST(batch_size, model_name),
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