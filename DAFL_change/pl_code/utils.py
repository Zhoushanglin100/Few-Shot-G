import os
from pickletools import optimize
from sched import scheduler
import warnings
import torch
import torch.nn.functional as F
import torchvision
import torchvision.models as tvmodels
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from model.resnet import *
from model.vgg_block import vgg_stock, vgg_bw, cfgs, split_block
from model.generator import Generator, GeneratorList


class GeneratorLR(_LRScheduler):

    def __init__(self, optimizer):
        super(GeneratorLR, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= 10: 
            lrs = [lr/10 for lr in self.base_lrs]
        else:
            lrs = self.base_lrs
        return lrs

class StudentLR(_LRScheduler):          # torch.optim.LambdaLR

    def __init__(self, optimizer, n_epochs_G, decay):
        self.n_epochs_G = n_epochs_G
        self.decay = decay
        super(StudentLR, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.n_epochs_G: # only train G
            lrs = [0 for _ in self.base_lrs]
        elif self.last_epoch < self.n_epochs_G+10: # warn up
            lrs = [lr * (self.last_epoch-self.n_epochs_G) / 10 for lr in self.base_lrs]
        else:
            lr_sq = ((self.last_epoch-self.n_epochs_G) // self.decay)+1
            lrs = [(0.977 ** lr_sq) * lr for lr in self.base_lrs]
        return lrs


def kdloss(y, teacher_scores):
    p = F.log_softmax(y, dim=1)
    q = F.softmax(teacher_scores, dim=1)
    l_kl = F.kl_div(p, q, reduction='sum')  / y.shape[0]
    return l_kl


### load teacher
def get_teacher(args):
    print(f"Dataset: {args.dataset}, teacher: {args.arch}")
    if args.dataset == "cifar10":
        if args.arch == "resnet34":
            # teacher = torch.load(args.teacher_dir + 'teacher_acc_95.3')
            teacher = ResNet34(num_classes=10)
            teacher.load_state_dict(torch.load(args.teacher_dir + "cifar10_resnet34_95.3.pth"))
        elif args.arch == "vgg16":
            teacher = vgg_stock(cfgs['vgg16'], args.dataset, 10)
            checkpoint = torch.load(args.teacher_dir + 'vgg16_CIFAR10_ckpt.pth')
            teacher.load_state_dict(checkpoint['net'])
    elif args.dataset == "cifar100":
        if args.arch == "resnet34":    
            teacher = ResNet34(num_classes=100)
            ckpt_teacher = torch.load(args.teacher_dir + "cifar100_resnet34.pth")    # 74.41%
            teacher.load_state_dict(ckpt_teacher['state_dict'])
        elif args.arch == "vgg16":
            teacher = vgg_stock(cfgs['vgg16'], args.dataset, 100)
            checkpoint = torch.load(args.teacher_dir + 'vgg16_CIFAR100_ckpt.pth')
            teacher.load_state_dict(checkpoint['net'])
    elif args.dataset == "tiny":
        teacher = ResNet34(num_classes=200)
        file_name = args.teacher_dir + "tinyimagenet_resnet34.pth"
        teacher.load_state_dict(torch.load(file_name))
    else:
        teacher = tvmodels.resnet34(pretrained=True)

    return teacher


### load student
def get_student(args):
    if args.dataset == 'cifar10':
        if args.arch_s == "resnet":
            net = ResNet18()
        elif args.arch_s == "vgg":
            net = vgg_bw(cfgs['vgg16-graft'], True, args.dataset, 10)
    elif args.dataset == 'cifar100':
        if args.arch_s == "resnet":
            net = ResNet18(num_classes=100)
        elif args.arch_s == "vgg":
            net = vgg_bw(cfgs['vgg16-graft'], True, args.dataset, 100)
    elif args.dataset == 'tiny':
        net = ResNet18(num_classes=200)

    return net


def save(epoch, model, optimizer, scheduler, save_path):
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()},
                save_path)


def load(load_path, model, optimizer, scheduler):
    assert os.path.exists(load_path)
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch'] + 1
    return epoch, model, optimizer, scheduler


if __name__ == '__main__':
    # y = torch.randn(20)
    # t_s = torch.randn(20)
    # print(kdloss(y, t_s))

    W = torch.randn(10, requires_grad=True)
    optimizer = torch.optim.SGD([W], lr=0.01)
    scheduler = StudentLR(optimizer, 3, 2)
    print(optimizer.param_groups)
    for i in range(5, 50):
        scheduler.step(epoch=i)
        print(scheduler.get_last_lr())
        # adjust_learning_rate(optimizer, 0.01, i, 3, 2)
