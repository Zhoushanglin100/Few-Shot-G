import torch
import torchvision
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset

# -----------------------------------------------------------

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
        torch.utils.data.dataset.Subset(train, np.where(target_train_idx == 1)[0]), 
        batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.dataset.Subset(test, np.where(target_test_idx == 1)[0]),
        batch_size=100)

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

### https://towardsdatascience.com/pytorch-ignite-classifying-tiny-imagenet-with-efficientnet-e5b1768e5e8f
def get_split_TinyImageNet(args, DATA_DIR, batch_size, start, end):

    start_class = start
    end_class = end

    TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
    VALID_DIR = os.path.join(DATA_DIR, 'val')

    transform_train = torchvision.transforms.Compose([
                                torchvision.transforms.Resize(256),               # Resize images to 256 x 256
                                torchvision.transforms.CenterCrop(224),           # Center crop image
                                torchvision.transforms.RandomHorizontalFlip(),
                                torchvision.transforms.ToTensor(),                 # Converting cropped images to tensors
                                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])

    transform_test = torchvision.transforms.Compose([
                                torchvision.transforms.Resize(256),               # Resize images to 256 x 256
                                torchvision.transforms.CenterCrop(224),           # Center crop image
                                torchvision.transforms.ToTensor(),                # Converting cropped images to tensors
                                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])

    train = torchvision.datasets.ImageFolder(TRAIN_DIR, transform=transform_train)

    # --------------
    # Create separate validation subfolders for the validation images based on
    # their labels indicated in the val_annotations txt file
    val_img_dir = os.path.join(VALID_DIR, 'images')

    # # Open and read val annotations text file
    # fp = open(os.path.join(VALID_DIR, 'val_annotations.txt'), 'r')
    # data = fp.readlines()

    # # Create dictionary to store img filename (word 0) and corresponding
    # # label (word 1) for every line in the txt file (as key value pair)
    # val_img_dict = {}
    # for line in data:
    #     words = line.split('\t')
    #     val_img_dict[words[0]] = words[1]
    # fp.close()

    # # Display first 10 entries of resulting val_img_dict dictionary
    # # {k: val_img_dict[k] for k in list(val_img_dict)[:10]}

    # # Create subfolders (if not present) for validation images based on label,
    # # and move images into the respective folders
    # for img, folder in val_img_dict.items():
    #     newpath = (os.path.join(val_img_dir, folder))
    #     if not os.path.exists(newpath):
    #         os.makedirs(newpath)
    #     if os.path.exists(os.path.join(val_img_dir, img)):
    #         os.rename(os.path.join(val_img_dir, img), os.path.join(newpath, img))

    test = torchvision.datasets.ImageFolder(val_img_dir, transform=transform_test)
    # --------------

    targets_train = torch.tensor(train.targets)
    target_train_idx = ((targets_train >= start_class) & (targets_train < end_class))
    targets_test = torch.tensor(test.targets)
    target_test_idx = ((targets_test >= start_class) & (targets_test < end_class))

    kwargs = {"pin_memory": True, "num_workers": 8}
    train_loader = torch.utils.data.DataLoader(torch.utils.data.dataset.Subset(train, np.where(target_train_idx == 1)[0]), 
                                                    batch_size=batch_size, shuffle=True, **kwargs) 
    test_loader = torch.utils.data.DataLoader(torch.utils.data.dataset.Subset(test, np.where(target_test_idx == 1)[0]), 
                                                    batch_size=100)    

    return train_loader, test_loader



def onehot(num_classes, y, device):
    y_hot = torch.zeros((len(y), num_classes)).to(device)
    for i,label in enumerate(y):
        y_hot[i][label] = 1

    return y_hot