import argparse
import numpy as np
import matplotlib.pyplot as plt

MNIST_PATH = '/content/drive/MyDrive/Github/ECS795P-Demo/MNIST_results'
MNISTFASHION_PATH = '/content/drive/MyDrive/Github/ECS795P-Demo/MNISTFashion_results'
CIFAR10_PATH = '/content/drive/MyDrive/Github/ECS795P-Demo/CIFAR10_results'

FILENAME = 'training_history_40.npy'

BY_DATASET = {
    'MNIST': {
        'VGG16': '{}/VGG16/{}'.format(MNIST_PATH, FILENAME),
        'ResNet101': '{}/ResNet101/{}'.format(MNIST_PATH, FILENAME),
        'ResNet101SE': '{}/ResNet101SE/{}'.format(MNIST_PATH, FILENAME),
        'ResidualAttention56': '{}/ResidualAttention56/{}'.format(MNIST_PATH, FILENAME),
        'ResNet101CBAM': '{}/ResNet101CBAM/{}'.format(MNIST_PATH, FILENAME),
    },
    'MNISTFashion': {
        'VGG16': '{}/VGG16/{}'.format(MNISTFASHION_PATH, FILENAME),
        'ResNet101': '{}/ResNet101/{}'.format(MNISTFASHION_PATH, FILENAME),
        'ResNet101SE': '{}/ResNet101SE/{}'.format(MNISTFASHION_PATH, FILENAME),
        'ResidualAttention56': '{}/ResidualAttention56/{}'.format(MNISTFASHION_PATH, FILENAME),
        'ResNet101CBAM': '{}/ResNet101CBAM/{}'.format(MNISTFASHION_PATH, FILENAME),
    },
    'CIFAR10': {
        'VGG16': '{}/VGG16/{}'.format(CIFAR10_PATH, FILENAME),
        'ResNet101': '{}/ResNet101/{}'.format(CIFAR10_PATH, FILENAME),
        'ResNet101SE': '{}/ResNet101SE/{}'.format(CIFAR10_PATH, FILENAME),
        'ResidualAttention56': '{}/ResidualAttention56/{}'.format(CIFAR10_PATH, FILENAME),
        'ResNet101CBAM': '{}/ResNet101CBAM/{}'.format(CIFAR10_PATH, FILENAME),
    },
}

def plotter():
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'CIFAR10', 'MNISTFashion'], help='Dataset')
    opt = parser.parse_args()

    print("==== Training Loss Over Epoch =====")
    train_loss(opt.dataset)
    test_loss(opt.dataset)
    print("==== Training Accuracy Over Epoch =====")
    train_accuracy(opt.dataset)
    test_accurcy(opt.dataset)

def train_loss(dataset, save=True, show=True):
    for model, path in BY_DATASET.get(dataset).items():
        print("Model {} ".format(model))
        train_history = np.load(path, allow_pickle=True)
        train_history = np.ndarray.tolist(train_history)
        x = range(len(train_history['train_avg_loss']))
        y1 = train_history['train_avg_loss']
        plt.plot(x, y1, label='Training Loss {}'.format(model))

    plt.xlabel('Epoch')
    plt.ylabel('Avg Loss')
    plt.title('Training Loss Over Epoch - {}'.format(dataset))
    plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig("Loss_Train_{}.png".format(dataset))
    if show:
        plt.show()
    else:
        plt.close()


def test_loss(dataset, save=True, show=True):
    for model, path in BY_DATASET.get(dataset).items():
        print("Model {} ".format(model))
        train_history = np.load(path, allow_pickle=True)
        train_history = np.ndarray.tolist(train_history)
        x = range(len(train_history['train_avg_loss']))
        y1 = train_history['test_avg_loss']
        plt.plot(x, y1, label='Testing Loss {}'.format(model))

    plt.xlabel('Epoch')
    plt.ylabel('Avg Loss')
    plt.title('Test Loss Over Epoch - {}'.format(dataset))
    plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig("Loss_Test_{}.png".format(dataset))
    if show:
        plt.show()
    else:
        plt.close()


def train_accuracy(dataset, save=True, show=True):
    for model, path in BY_DATASET.get(dataset).items():
        print("Model {} ".format(model))
        train_history = np.load(path, allow_pickle=True)
        train_history = np.ndarray.tolist(train_history)
        x = range(len(train_history['train_accuracy']))
        y1 = train_history['train_accuracy']
        plt.plot(x, y1, label='Training Accuracy {}'.format(model))
        print("Final Train for Model {} is {}".format(model, train_history['train_accuracy'][-1]))

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Over Epoch - {}'.format(dataset))
    plt.legend(bbox_to_anchor=(1, 1), loc=4, borderaxespad=0)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig("Accuracy_Train_{}.png".format(dataset))
    if show:
        plt.show()
    else:
        plt.close()


def test_accurcy(dataset, save=True, show=True):
    for model, path in BY_DATASET.get(dataset).items():
        print("Model {} ".format(model))
        train_history = np.load(path, allow_pickle=True)
        train_history = np.ndarray.tolist(train_history)
        x = range(len(train_history['test_accuracy']))
        y1 = train_history['test_accuracy']
        plt.plot(x, y1, label='Testing Accuracy {}'.format(model))
        print("Final Test Validation Accuracy for Model {} is {}".format(model, train_history['test_accuracy'][-1]))

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test test_accuracy Over Epoch - {}'.format(dataset))
    plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig("Accuracy_Test_{}.png".format(dataset))
    if show:
        plt.show()
    else:
        plt.close()

if __name__ == '__main__':
    plotter()
