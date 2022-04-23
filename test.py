import argparse
import torch
import torchvision
from utils import get_model, get_dataset, load_image_from_tensor
from main import test_loop
import torch.nn as nn


def test_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_path', )
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size')
    parser.add_argument('--dataset', type=str, default='MNISTFashion', choices=['MNIST', 'CIFAR10', 'MNISTFashion'], help='Training/Test Dataset')
    parser.add_argument('--model_name', type=str, default='VGG13', choices=['VGG13', 'VGG16'], help='Name of architecture')
    parser.add_argument('--model_path', type=str, default='C:/Users/jsan/PycharmProjects/CV_CW3/MNISTFashion_results/VGG13/model_VGG13_epoch50.pth', help='Full path to the model')
    parser.add_argument('--full_test', type=bool, default=False, help='Run full test loop on model to get accuracy and loss from validation dataset')
    opt = parser.parse_args()

    # Get dataset
    dataset = get_dataset(opt.dataset, opt.model_name, opt.batch_size)
    test_loader = dataset.get_test_loader()

    test_iter = iter(test_loader)

    images, labels = test_iter.next()
    # Save Images
    save_path = "{}/test_img.png".format(dataset.get_results_model_dir())
    load_image_from_tensor(torchvision.utils.make_grid(images), save_path=save_path)

    # Load Ground Truth Laels
    id_to_class_map = dataset.get_id_to_class_mapping()
    ground_truth_labels = "  |  ".join([id_to_class_map[label.item()] for label in labels])

    # initialise the device for training, if gpu is available, device = 'cuda', else: device = 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = get_model(opt.model_name, device, opt.dataset)
    model.load_state_dict(torch.load(opt.model_path))

    # Get predictions
    outputs = model(images.to(device=device))
    _, predicted = torch.max(outputs, 1)
    predicted_labels = "  |  ".join([id_to_class_map[label.item()] for label in predicted])

    print("--- Ground Truth Labels ---")
    print(ground_truth_labels)
    print("--- Predicted Labels ---")
    print(predicted_labels)

    if opt.full_test:
        print("--- Accuracy and Loss from Validation Set ---")
        criterion = nn.CrossEntropyLoss()
        test_loop(test_loader, model, criterion, device)

if __name__ == '__main__':
    test_model()