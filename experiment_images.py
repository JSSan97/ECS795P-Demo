import argparse
import torch
import torchvision
from utils import get_model, get_dataset, load_image_from_tensor
from main import test_loop
import torch.nn as nn

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
    'MNISTFashion': {
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


def test_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_path', )
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size')
    parser.add_argument('--dataset', type=str, default='MNISTFashion', choices=['MNIST', 'CIFAR10', 'MNISTFashion'], help='Training/Test Dataset')
    # parser.add_argument('--model_name', type=str, default='VGG16', choices=['VGG16', 'ResNet101', 'ResNet101SE', 'ResNet101CBAM', 'ResidualAttention56'], help='Name of architecture')
    # parser.add_argument('--model_path', type=str, default='C:/Users/jsan/PycharmProjects/CV_CW3/MNISTFashion_results/VGG16/model_VGG16_epoch40.pth', help='Full path to the model')
    # parser.add_argument('--full_test', type=bool, default=False, help='Run full test loop on model to get accuracy and loss from validation dataset')
    opt = parser.parse_args()

    # Get dataset
    dataset = get_dataset(opt.dataset, "VGG16", opt.batch_size)
    test_loader = dataset.get_test_loader()

    test_iter = iter(test_loader)
    images, labels = test_iter.next()

    print("--- Input Images ---")
    save_path = "{}/test_img.png".format(dataset.get_results_model_dir())
    load_image_from_tensor(torchvision.utils.make_grid(images), save_path=save_path)

    # Load Ground Truth Labels
    id_to_class_map = dataset.get_id_to_class_mapping()
    ground_truth_labels = "  |  ".join([id_to_class_map[label.item()] for label in labels])

    print("--- Ground Truth Labels ---")
    print(ground_truth_labels)

    # initialise the device for training, if gpu is available, device = 'cuda', else: device = 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for model, path in BY_DATASET.get(opt.dataset).items():
        # Load model
        model = get_model(model, device, dataset)
        model.load_state_dict(torch.load(path))

        # Get predictions
        outputs = model(images.to(device=device))
        _, predicted = torch.max(outputs, 1)
        predicted_labels = "  |  ".join([id_to_class_map[label.item()] for label in predicted])

        print("--- Model {} Predicted Labels ---".format(model))
        print(predicted_labels)

    # if opt.full_test:
    #     print("--- Accuracy and Loss from Validation Set ---")
    #     criterion = nn.CrossEntropyLoss()
    #     test_loop(test_loader, model, criterion, device)

if __name__ == '__main__':
    test_model()