import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

HISTORY_PATH = {
    'VGG16' : {
        'MNIST' : 'MNIST_results\VGG16\\training_history_30.npy',
        'MNISTFashion' : 'MNISTFashion_results\VGG16\\training_history_30.npy',
        'CIFAR10' : 'CIFAR10_results\VGG16\\training_history_30.npy',
    },
    'VGG13': {
        'MNIST': 'MNIST_results\VGG13\\training_history_30.npy',
        'MNISTFashion': 'MNISTFashion_results\VGG13\\training_history_30.npy',
        'CIFAR10': 'CIFAR10_results\VGG13\\training_history_30.npy',
    },
    'ResNet101': {
        'MNIST': 'MNIST_results\ResNet101\\training_history_30.npy',
        'MNISTFashion': 'MNISTFashion_results\\ResNet101\training_history_30.npy',
        'CIFAR10': 'CIFAR10_results\\ResNet101\training_history_30.npy',
    },
    'ResNet50': {
        'MNIST': 'MNIST_results\ResNet50\\training_history_30.npy',
        'MNISTFashion': 'MNISTFashion_results\ResNet50\\training_history_30.npy',
        'CIFAR10': 'CIFAR10_results\ResNet50\\training_history_30.npy',
    }
}

def plotter(save=True, show=True):
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='VGG13', choices=['VGG13', 'VGG16', 'ResNet50', 'ResNet101', 'GoogLeNet'], help='Name of architecture')
    opt = parser.parse_args()

    current_path = os.path.dirname(os.path.realpath(__file__))

    for dataset, rel_path in HISTORY_PATH.get(opt.model_name).items():
        print('{}\{}'.format(current_path, rel_path))
        train_history = np.load('{}\{}'.format(current_path, rel_path), allow_pickle=True)
        train_history = np.ndarray.tolist(train_history)
        x = train_history['time']
        y1 = train_history['train_avg_loss']
        y2 = train_history['test_avg_loss']
        plt.plot(x, y1, label='Training Loss {}'.format(dataset))
        plt.plot(x, y2, label='Testing Loss {}'.format(dataset))


    plt.xlabel('Time')
    plt.ylabel('Loss')
    plt.title('Test & Training Loss Over Time')
    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig("Loss_{}".format(opt.model_name))
    if show:
        plt.show()
    else:
        plt.close()



if __name__ == '__main__':
    plotter()
