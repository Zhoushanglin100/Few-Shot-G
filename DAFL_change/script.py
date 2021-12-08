import torch
import torchvision
import numpy as np


def get_split_cifar100(args, batch_size=32, start=0, end=50):
    shuffle = False
    # convention: tasks starts from 1 not 0 !
    # task_id = 1 (i.e., first task) => start_class = 0, end_class = 4
    # start_class = (task_id - 1) * 5
    start_class = start
    # end_class = task_id * 5
    end_class = end

    # transforms = torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    # Normalize test set same as training set without augmentation
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    train = torchvision.datasets.CIFAR100(args.data, train=True, download=True, transform=transform_train)
    test = torchvision.datasets.CIFAR100(args.data, train=False, download=True, transform=transform_test)

    targets_train = torch.tensor(train.targets)
    target_train_idx = ((targets_train >= start_class) & (targets_train < end_class))

    targets_test = torch.tensor(test.targets)
    target_test_idx = ((targets_test >= start_class) & (targets_test < end_class))

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.dataset.Subset(train, np.where(target_train_idx == 1)[0]), batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.dataset.Subset(test, np.where(target_test_idx == 1)[0]),
                                              batch_size=batch_size)

    return train_loader, test_loader


def get_split_cifar10(args, batch_size, start, end):

    # convention: tasks starts from 1 not 0 !
    # task_id = 1 (i.e., first task) => start_class = 0, end_class = 4
    # start_class = (task_id - 1) * 5
    start_class = start
    # end_class = task_id * 5
    end_class = end

    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    train = torchvision.datasets.CIFAR10(args.data, train=True, download=True, transform=transform_train)
    test = torchvision.datasets.CIFAR10(args.data, train=False, download=True, transform=transform_test)

    targets_train = torch.tensor(train.targets)
    target_train_idx = ((targets_train >= start_class) & (targets_train < end_class))
    # print("!!! target_test_idx", target_train_idx)

    targets_test = torch.tensor(test.targets)
    target_test_idx = ((targets_test >= start_class) & (targets_test < end_class))
    # print("!!! target_test_idx", target_test_idx)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.dataset.Subset(train, np.where(target_train_idx == 1)[0]), 
        batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.dataset.Subset(test, np.where(target_test_idx == 1)[0]), 
        batch_size=100)
    # print("!!! train_loader", train_loader)
    # print("!!! test_loader", test_loader)

    return train_loader, test_loader



def onehot(num_classes, y, device):
    y_hot = torch.zeros((len(y), num_classes)).to(device)
    for i,label in enumerate(y):
        y_hot[i][label] = 1

    return y_hot