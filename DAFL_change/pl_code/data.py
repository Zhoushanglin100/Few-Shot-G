import torchvision
import torch
from torch.utils.data import Dataset, DataLoader


class BaseDataLoader():
    def __init__(self):
        self.trainset = None
        self.testset = None

    def _transform(self):
        raise NotImplementedError

    def build_trainloader(self, batch_size=128):
        return DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    def build_testloader(self, batch_size=100):
        return DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=4)


class CifarDataLoader(BaseDataLoader):
    def __init__(self, name='cifar10', cache_path='../cache/data/'):
        transform_train, transform_test = self._transform()
        if name == 'cifar10':
            self.trainset = torchvision.datasets.CIFAR10(cache_path, train=True, download=True, transform=transform_train)
            self.testset = torchvision.datasets.CIFAR10(cache_path, train=False, download=True, transform=transform_test)
        elif name == 'cifar100':
            self.trainset = torchvision.datasets.CIFAR100(cache_path, train=True, download=True, transform=transform_train)
            self.testset = torchvision.datasets.CIFAR100(cache_path, train=False, download=True, transform=transform_test)

    def _transform(self):
        transform_train = torchvision.transforms.Compose([
                                    torchvision.transforms.RandomCrop(32, padding=4),
                                    torchvision.transforms.RandomHorizontalFlip(),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
                                    ])
        transform_test = torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                    ])
        return transform_train, transform_test


class ImagenetDataLoader(BaseDataLoader):
    def __init__(self, imagenet_path='/data/imagenet/train'):
        transform = self._transform()
        self.trainset = torchvision.datasets.ImageFolder(imagenet_path, transform=transform)

    def _transform(self):
        transform_train = torchvision.transforms.Compose([
                                    torchvision.transforms.Resize(size = (32,32)),
                                    torchvision.transforms.RandomHorizontalFlip(),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])
        return transform_train

    def build_trainloader(self, batch_size=128):
        return DataLoader(self.trainset, batch_size=batch_size, shuffle=False, num_workers=4)
    

# class TinyimagenetDataLoader(BaseDataLoader):
#     def build_testloader(self, args, batch_size=100):
#         DATA_DIR = "/data/tiny-imagenet-200"
#         _, data_test_loader = get_split_TinyImageNet(args, DATA_DIR, batch_size, 0, 200)
#         return data_test_loader

