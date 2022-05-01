import argparse
import numpy as np
import matplotlib.pyplot as plt

MNIST_PATH = '/content/drive/MyDrive/Github/ECS795P-Demo/MNIST_results'
MNISTFASHION_PATH = '/content/drive/MyDrive/Github/ECS795P-Demo/MNISTFashion_results'
CIFAR10_PATH = '/content/drive/MyDrive/Github/ECS795P-Demo/CIFAR10_results'

BY_DATASET = {
    'MNIST': {
        'VGG16': '{}/VGG16/model_VGG16_epoch40.pth'.format(MNIST_PATH),
        'ResNet101': '{}/ResNet101/model_ResNet101_epoch40.pth'.format(MNIST_PATH),
        'ResNet101SE': '{}/ResNet101SE/model_ResNet101SE_epoch40.pth'.format(MNIST_PATH),
        'ResidualAttention56': '{}/ResidualAttention56/model_ResidualAttention56_epoch40.pth'.format(MNIST_PATH),
        'ResNet101CBAM': '{}/ResNet101CBAM/model_ResNet101CBAM_epoch40.pth'.format(MNIST_PATH),
    },
    'MNISTFASHION': {
        'VGG16': '{}/VGG16/model_VGG16_epoch40.pth'.format(MNISTFASHION_PATH),
        'ResNet101': '{}/ResNet101/model_ResNet101_epoch40.pth'.format(MNISTFASHION_PATH),
        'ResNet101SE': '{}/ResNet101SE/model_ResNet101SE_epoch40.pth'.format(MNISTFASHION_PATH),
        'ResidualAttention56': '{}/ResidualAttention56/model_ResidualAttention56_epoch40.pth'.format(MNISTFASHION_PATH),
        'ResNet101CBAM': '{}/ResNet101CBAM/model_ResNet101CBAM_epoch40.pth'.format(MNISTFASHION_PATH),
    },
    'CIFAR10': {
        'VGG16': '{}/VGG16/model_VGG16_epoch40.pth'.format(CIFAR10_PATH),
        'ResNet101': '{}/ResNet101/model_ResNet101_epoch40.pth'.format(CIFAR10_PATH),
        'ResNet101SE': '{}/ResNet101SE/model_ResNet101SE_epoch40.pth'.format(CIFAR10_PATH),
        'ResidualAttention56': '{}/ResidualAttention56/model_ResidualAttention56_epoch40.pth'.format(CIFAR10_PATH),
        'ResNet101CBAM': '{}/ResNet101CBAM/model_ResNet101CBAM_epoch40.pth'.format(CIFAR10_PATH),
    },
}

def plotter(save=True, show=True):
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'CIFAR10', 'MNISTFashion'], help='Dataset')
    opt = parser.parse_args()

    for model, path in BY_DATASET.get(opt.dataset).items():
        train_history = np.load(path, allow_pickle=True)
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
