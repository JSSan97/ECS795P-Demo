import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os
import ssl

class Dataset():
    def __init__(self, train_loader, test_loader, test_dataset, results_model_dir):
        self.train_loader = train_loader
        self.test_dataset = test_dataset
        self.test_loader = test_loader
        self.results_model_dir = results_model_dir

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

        transform = get_input_transform(model_name)

        train_data = datasets.MNIST(root=self.train_dir, train=True, transform=transform, download=True)
        self.train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

        test_data = datasets.MNIST(root=self.test_dir, train=False, transform=transform, download=True)
        self.test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

        super().__init__(self.train_loader, self.test_loader, test_data, self.results_model_dir)

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

        transform = get_input_transform(model_name)

        # SSL error from downloading CIFAR10 dataset, need to have this line
        ssl._create_default_https_context = ssl._create_unverified_context

        train_data = datasets.CIFAR10(root=self.train_dir, train=True, transform=transform, download=True)
        self.train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

        test_data = datasets.CIFAR10(root=self.test_dir, train=False, transform=transform, download=True)
        self.test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

        super().__init__(self.train_loader, self.test_loader, test_data, self.results_model_dir)


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

        transform = get_input_transform(model_name)

        train_data = datasets.FashionMNIST(root=self.train_dir, train=True, transform=transform, download=True)
        self.train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

        test_data = datasets.FashionMNIST(root=self.test_dir, train=False, transform=transform, download=True)
        self.test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

        super().__init__(self.train_loader, self.test_loader, test_data, self.results_model_dir)

def get_input_transform(model_name):
    if model_name == "VGG13" or model_name == "VGG16":
        transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5,), std=(0.5,))])

    return transform
