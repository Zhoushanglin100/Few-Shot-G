import torch
import torchvision
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torchvision.models as models
import time
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input, target = input.cuda(), target.cuda()
        # print(input.shape, target.shape)
        
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


# DATA_DIR = "/data/tiny-imagenet-200"
# start_class = 0
# end_class = 200
# batch_size = 32

# TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
# VALID_DIR = os.path.join(DATA_DIR, 'val')

# transform_train = torchvision.transforms.Compose([
#                             torchvision.transforms.Resize(256),               # Resize images to 256 x 256
#                             torchvision.transforms.CenterCrop(224),           # Center crop image
#                             torchvision.transforms.RandomHorizontalFlip(),
#                             torchvision.transforms.ToTensor(),                 # Converting cropped images to tensors
#                             torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                             ])

# transform_test = torchvision.transforms.Compose([
#                             torchvision.transforms.Resize(256),               # Resize images to 256 x 256
#                             torchvision.transforms.CenterCrop(224),           # Center crop image
#                             torchvision.transforms.ToTensor(),                 # Converting cropped images to tensors
#                             torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                             ])

# train = torchvision.datasets.ImageFolder(TRAIN_DIR, transform=transform_train)
# kwargs = {"pin_memory": True, "num_workers": 4}

# # --------------
# # Create separate validation subfolders for the validation images based on
# # their labels indicated in the val_annotations txt file
# val_img_dir = os.path.join(VALID_DIR, 'images')

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
# {k: val_img_dict[k] for k in list(val_img_dict)[:10]}

# # Create subfolders (if not present) for validation images based on label,
# # and move images into the respective folders
# for img, folder in val_img_dict.items():
#     newpath = (os.path.join(val_img_dir, folder))
#     if not os.path.exists(newpath):
#         os.makedirs(newpath)
#     if os.path.exists(os.path.join(val_img_dir, img)):
#         os.rename(os.path.join(val_img_dir, img), os.path.join(newpath, img))
# # --------------

# test = torchvision.datasets.ImageFolder(val_img_dir, transform=transform_test)

# targets_train = torch.tensor(train.targets)
# target_train_idx = ((targets_train >= start_class) & (targets_train < end_class))
# targets_test = torch.tensor(test.targets)
# target_test_idx = ((targets_test >= start_class) & (targets_test < end_class))

# train_loader = torch.utils.data.DataLoader(torch.utils.data.dataset.Subset(train, np.where(target_train_idx == 1)[0]), 
#                                                 batch_size=batch_size, shuffle=True, num_workers=8)
    
# test_loader = torch.utils.data.DataLoader(torch.utils.data.dataset.Subset(test, np.where(target_test_idx == 1)[0]), 
#                                                 batch_size=100)    


# train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, **kwargs)



# -----------------

# # Data loading code
# traindir = os.path.join("/data/imagenet", 'train')
# valdir = os.path.join("/data/imagenet", 'val')
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                     std=[0.229, 0.224, 0.225])

# train_dataset = datasets.ImageFolder(
#     traindir,
#     transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         normalize,
#     ]))

# train_sampler = None

# train_loader = torch.utils.data.DataLoader(
#     train_dataset, batch_size=32, shuffle=(train_sampler is None),
#     num_workers=8, pin_memory=True, sampler=train_sampler)

# test_loader = torch.utils.data.DataLoader(
#     datasets.ImageFolder(valdir, transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         normalize,
#     ])),
#     batch_size=32, shuffle=False,
#     num_workers=8, pin_memory=True)


# -----------------

# Define main data directory
DATA_DIR = '/data/tiny-imagenet-200' # Original images come in shapes of [3,64,64]

# Define training and validation data paths
TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
VALID_DIR = os.path.join(DATA_DIR, 'val')

def generate_dataloader(data, name, transform):
    if data is None: 
        return None
    
    # Read image files to pytorch dataset using ImageFolder, a generic data 
    # loader where images are in format root/label/filename
    # See https://pytorch.org/vision/stable/datasets.html
    if transform is None:
        dataset = datasets.ImageFolder(data, transform=transforms.ToTensor())
    else:
        dataset = datasets.ImageFolder(data, transform=transform)

    # Set options for device
    kwargs = {"pin_memory": True, "num_workers": 1}

    
    # Wrap image dataset (defined above) in dataloader 
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                        shuffle=(name=="train"), 
                        **kwargs)
    
    return dataloader


val_img_dir = os.path.join(VALID_DIR, 'images')

fp = open(os.path.join(VALID_DIR, 'val_annotations.txt'), 'r')
data = fp.readlines()

val_img_dict = {}
for line in data:
    words = line.split('\t')
    val_img_dict[words[0]] = words[1]
fp.close()

for img, folder in val_img_dict.items():
    newpath = (os.path.join(val_img_dir, folder))
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    if os.path.exists(os.path.join(val_img_dir, img)):
        os.rename(os.path.join(val_img_dir, img), os.path.join(newpath, img))

preprocess_transform_pretrain = transforms.Compose([
                transforms.Resize(256), # Resize images to 256 x 256
                transforms.CenterCrop(224), # Center crop image
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # Converting cropped images to tensors
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
])

batch_size = 64

train_loader_pretrain = generate_dataloader(TRAIN_DIR, "train",
                                  transform=preprocess_transform_pretrain)

test_loader = generate_dataloader(val_img_dir, "val",
                                 transform=preprocess_transform_pretrain)

# ------------------

model = models.resnet34(pretrained=True)
model.cuda()


import torch.nn as nn

criterion = nn.CrossEntropyLoss().cuda()
validate(test_loader, model, criterion)