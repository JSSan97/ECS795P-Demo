import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os
import ssl

class Dataset():
    def __init__(self, dataset_name, train_loader, test_loader, test_dataset, results_model_dir):
        self.dataset_name = dataset_name
        self.train_loader = train_loader
        self.test_dataset = test_dataset
        self.test_loader = test_loader
        self.results_model_dir = results_model_dir

    def get_number_of_classes(self):
        return len(self.test_dataset.class_to_idx)

    def get_dataset_name(self):
        return self.dataset_name

    def get_class_to_id_mapping(self):
        return self.test_dataset.class_to_idx

    def get_id_to_class_mapping(self):
        return dict((v, k) for k, v in self.test_dataset.class_to_idx.items())

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader

    def get_results_model_dir(self):
        return self.results_model_dir

class MNISTDigits(Dataset):
    def __init__(self, batch_size, model_name):
        self.train_dir = './MNIST_data_train/'
        self.test_dir = './MNIST_data_test/'
        self.results_dir = './MNIST_results/'
        self.results_model_dir = './MNIST_results/{}'.format(model_name)

        # create folder if not exist
        if not os.path.exists(self.results_dir):
            os.mkdir(self.results_dir)
        if not os.path.exists(self.results_model_dir):
            os.mkdir(self.results_model_dir)

        transform_train, transform_test = get_input_transform('MNIST')

        train_data = datasets.MNIST(root=self.train_dir, train=True, transform=transform_train, download=True)
        self.train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

        test_data = datasets.MNIST(root=self.test_dir, train=False, transform=transform_test, download=True)
        self.test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

        super().__init__('MNIST', self.train_loader, self.test_loader, test_data, self.results_model_dir)

class CIFAR10(Dataset):
    def __init__(self, batch_size, model_name):
        self.train_dir = './CIFAR10_data_train/'
        self.test_dir = './CIFAR10_data_test/'
        self.results_dir = './CIFAR10_results/'
        self.results_model_dir = './CIFAR10_results/{}'.format(model_name)

        # create folder if not exist
        if not os.path.exists(self.results_dir):
            os.mkdir(self.results_dir)
        if not os.path.exists(self.results_model_dir):
            os.mkdir(self.results_model_dir)

        transform_train, transform_test = get_input_transform('CIFAR10')

        # SSL error from downloading CIFAR10 dataset, need to have this line
        ssl._create_default_https_context = ssl._create_unverified_context

        train_data = datasets.CIFAR10(root=self.train_dir, train=True, transform=transform_train, download=True)
        self.train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

        test_data = datasets.CIFAR10(root=self.test_dir, train=False, transform=transform_test, download=True)
        self.test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

        super().__init__('CIFAR10', self.train_loader, self.test_loader, test_data, self.results_model_dir)


class FashionMNIST(Dataset):
    def __init__(self, batch_size, model_name):
        self.train_dir = './MNISTFashion_data_train/'
        self.test_dir = './MNISTFashion_data_test/'
        self.results_dir = './MNISTFashion_results/'
        self.results_model_dir = './MNISTFashion_results/{}'.format(model_name)

        # create folder if not exist
        if not os.path.exists(self.results_dir):
            os.mkdir(self.results_dir)
        if not os.path.exists(self.results_model_dir):
            os.mkdir(self.results_model_dir)

        transform_train, transform_test = get_input_transform('MNISTFashion')

        train_data = datasets.FashionMNIST(root=self.train_dir, train=True, transform=transform_train, download=True)
        self.train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

        test_data = datasets.FashionMNIST(root=self.test_dir, train=False, transform=transform_test, download=True)
        self.test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

        super().__init__('MNISTFashion', self.train_loader, self.test_loader, test_data, self.results_model_dir)

def get_input_transform(dataset_name):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if dataset_name == 'MNISTFashion' or dataset_name == 'MNIST':
        normalize = transforms.Normalize(mean=(0.5,), std=(0.5,))

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    return transform_train, transform_test
