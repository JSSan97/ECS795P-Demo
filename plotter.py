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
    'MNISTFASHION': {
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

def plotter(save=True, show=True):
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'CIFAR10', 'MNISTFashion'], help='Dataset')
    opt = parser.parse_args()

    for model, path in BY_DATASET.get(opt.dataset).items():
        train_history = np.load(path, allow_pickle=True)
        print(train_history)
        train_history = np.ndarray.tolist(train_history)
        x = range(len(train_history['train_avg_loss']))
        y1 = train_history['train_avg_loss']
        y2 = train_history['test_avg_loss']
        plt.plot(x, y1, label='Training Loss {}'.format(model))
        plt.plot(x, y2, label='Testing Loss {}'.format(model))

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Test & Training Loss Over Epoch')
    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig("Loss_{}".format(opt.dataset))
    if show:
        plt.show()
    else:
        plt.close()

if __name__ == '__main__':
    plotter()
