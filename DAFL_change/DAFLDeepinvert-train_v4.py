#Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import os
from model.lenet import LeNet5
import resnet as resnet

import torch
from torch.autograd import Variable
from torchvision.datasets.mnist import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 
import argparse

import torch.nn as nn
import torch.nn.functional as F
import random

try:
    import wandb
    has_wandb = True
except ImportError: 
    has_wandb = False

# has_wandb = False

###########################################

parser = argparse.ArgumentParser(description='train-teacher-network')

# Basic model parameters.
parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST','cifar10','cifar100'])
parser.add_argument('--data', type=str, default='cache/data/')
parser.add_argument('--output_dir', type=str, default='cache/models/')
parser.add_argument('--teacher_dir', type=str, default='cache/models/')
parser.add_argument('--ext', type=str, default='')
parser.add_argument('--resume', action='store_true', default=False,
                    help='whether to resume from ckpt')

parser.add_argument('--n_epochs_G', type=int, default=50, help='number of epochs of training generator')
parser.add_argument('--n_epochs', type=int, default=2000, help='number of epochs of training total')
parser.add_argument('--fix_G', action='store_true', default=False,
                    help='whether stop train generator after start training student')

parser.add_argument('--batch_size', type=int, default=256, help='size of the batches')
parser.add_argument('--lr_G', type=float, default=0.2, help='learning rate of generator')
parser.add_argument('--lr_S', type=float, default=0.06, help='learning rate of student')
parser.add_argument('--decay', type=float, default=5, help='decay of learning rate')

parser.add_argument('--latent_dim', type=int, default=1000, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--oh', type=float, default=1, help='one hot loss')
parser.add_argument('--ie', type=float, default=5, help='information entropy loss')
parser.add_argument('--a', type=float, default=0.1, help='activation loss')

args = parser.parse_args()


if has_wandb:
    id = "{}-bz{}-{}-ld{}-eN{}-eG{}-lrG{}-lrS{}".format(args.dataset,
                                                            args.batch_size, args.fix_G, args.latent_dim,
                                                            args.n_epochs, args.n_epochs_G,
                                                            args.lr_G, args.lr_S)
    wandb.init(project='few-shot_multi', entity='zhoushanglin100', config=args, resume="allow", id=id)
    wandb.config.update(args)

img_shape = (args.channels, args.img_size, args.img_size)

os.makedirs(args.output_dir, exist_ok=True)  

acc = 0
acc_best = 0

start_epoch = 1
# ------------------------------------------------
### add deepinversion
class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]

        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        # print("output", mean.shape, var.shape)
        # print("batch", module.running_mean.data.shape, module.running_var.data.shape)
        # print("=====")

        # forcing mean and variance to match between two distributions
        # other ways might work better, e.g. KL divergence
        r_feature = torch.norm(module.running_var.data.type(var.type()) - var, 2) + torch.norm(
            module.running_mean.data.type(mean.type()) - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()
### end deepinversion

# ------------------------------------------------
### add gen
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = args.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(args.latent_dim, 128*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, args.channels, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(args.channels, affine=False)
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        return img

generator = Generator().cuda()
### end gen

# ---------------
def test_teacher(teacher):
    teacher.eval()
    total_correct = 0
    avg_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            output = teacher(images)
            avg_loss += criterion(output, labels).sum()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
 
    avg_loss /= len(data_test)
    acc = float(total_correct) / len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), acc))

# ------------------------------------------------
### Create dataset

if args.dataset == 'MNIST':
    
    data_train = MNIST(args.data,
                       transform=transforms.Compose([
                           transforms.Resize((32, 32)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                           ]))
    data_test = MNIST(args.data,
                      train=False,
                      transform=transforms.Compose([
                          transforms.Resize((32, 32)),
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                          ]))

    data_train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=8)
    data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)

    net = LeNet5().cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
if args.dataset == 'cifar10':
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data_train = CIFAR10(args.data, download=True,
                       transform=transform_train)
    data_test = CIFAR10(args.data,
                      train=False, download=True,
                      transform=transform_test)

    data_train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=8)
    data_test_loader = DataLoader(data_test, batch_size=100, num_workers=0)

    #net = resnet.ResNet34().cuda()
    net = resnet.ResNet18().cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    
    optimizer_S = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    # optimizer_S = torch.optim.SGD(net.parameters(), lr=args.lr_S, momentum=0.9, weight_decay=5e-4)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr_G)
    
if args.dataset == 'cifar100':
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data_train = CIFAR100(args.data,
                       transform=transform_train)
    data_test = CIFAR100(args.data,
                      train=False,
                      transform=transform_test)
                      
    data_train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    data_test_loader = DataLoader(data_test, batch_size=100, num_workers=0)
    net = resnet.ResNet18(num_classes=100).cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    
    optimizer_S = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    # optimizer_S = torch.optim.SGD(net.parameters(), lr=args.lr_S, momentum=0.9, weight_decay=5e-4)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr_G)

# ----------------------------------------------------------
if args.dataset == 'cifar10':
    teacher = torch.load(args.teacher_dir + 'teacher_acc_95.3').cuda()
if args.dataset == 'cifar100':
    # teacher = resnet_1.resnet34(num_classes=100).cuda()
    teacher = resnet.ResNet34(num_classes=100).cuda()
    ckpt_teacher = torch.load("cache/pretrained/cifar100_resnet34.pth")   # 74.41%
    teacher.load_state_dict(ckpt_teacher['state_dict'])

teacher.eval()
teacher = nn.DataParallel(teacher)
generator = nn.DataParallel(generator)
# deepinversion

# test_teacher(teacher)

# -------------------------------------
save_path = 'cache/ckpts_'+args.dataset
if not os.path.exists(save_path):
    os.makedirs(save_path)

    #    torch.save({
    #         'epoch': epoch,
    #         'S_state_dict': net.state_dict(),
    #         'S_optimizer_state_dict': optimizer_S.state_dict(),
    #         'G_state_dict':generator.state_dict(),
    #         'G_optimizer_state_dict':optimizer_G.state_dict()}, 
    #         save_path+"/"+save_name)

if args.resume:
    load_name = "{}_bz{}_{}_ld{}_eN{}_eG{}_lrG{}_lrS{}.pth".format(args.dataset, 
                                                                    args.batch_size, args.fix_G, args.latent_dim,
                                                                    args.n_epochs, args.n_epochs_G,
                                                                    args.lr_G, args.lr_S)
    if os.path.exists(save_path+'/'+load_name):
        checkpoint = torch.load(save_path+'/'+load_name)
        net.load_state_dict(checkpoint['S_state_dict'])
        optimizer_S.load_state_dict(checkpoint['S_optimizer_state_dict'])
        generator.load_state_dict(checkpoint['G_state_dict'])
        optimizer_G.load_state_dict(checkpoint['G_optimizer_state_dict'])
        resume_epoch = checkpoint['epoch']
        start_epoch = resume_epoch+1
    

    
# ------------------------------------------------
### adjust learning rate

def adjust_learning_rate(args, optimizer, epoch):
    """
    Learning rate for student
    For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    """ 
    if epoch < 10:
        lr = 0
    elif epoch < 160:
        lr = 0.06
    elif epoch < 320:
        lr = 0.006
    else:
        lr = 0.00006
    
    """ 
    if epoch < args.n_epochs_G: # only train G
        lr = 0
    elif epoch < args.n_epochs_G+10: # warn up
        lr = args.lr_S * (epoch-args.n_epochs_G)/ 10
    else:
        lr = args.lr_S # lr decay
        lr_sq = ((epoch-args.n_epochs_G) // args.decay)+1
        lr = (0.977 ** lr_sq) * lr
    

    # if epoch < args.n_epochs_G: # only train G
    #     lr = 0
    # elif epoch < args.n_epochs_G*1.5: # warn up
    #     lr = args.lr_S * (epoch-args.n_epochs_G)/ 25
    # else:
    #     lr = args.lr_S # lr decay
    #     lr_sq = ((epoch-args.n_epochs_G)// 25)+1

    #     lr = (0.977 ** lr_sq) * lr
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if has_wandb:
        wandb.log({"lr/lr_S": lr})

def adjust_learning_rate_G(args, optimizer, epoch):
    """
    Learning rate for generator
    """
    if epoch >= 10:
        lr = args.lr_G / 10
    else:
        lr = args.lr_G

    # if epoch < args.n_epochs_G*(1/10):
    #     lr = args.lr_G
    # else:
    #     lr = args.lr_G / 10

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if has_wandb:
        wandb.log({"lr/lr_G": lr})

# ------------------------------------------------

## Create hooks for feature statistics catching
loss_r_feature_layers = []
# for module in teacher.modules():
for name, module in teacher.named_modules():
    if isinstance(module, nn.BatchNorm2d):
        loss_r_feature_layers.append(DeepInversionFeatureHook(module))
# setting up the range for jitter
lim_0, lim_1 = 2, 2
# end deepinversion

def kdloss(y, teacher_scores):
    p = F.log_softmax(y, dim=1)
    q = F.softmax(teacher_scores, dim=1)
    l_kl = F.kl_div(p, q, size_average=False)  / y.shape[0]
    return l_kl


# --------------------------------------------------------

def train(epoch, args):
    if args.dataset != 'MNIST':
        adjust_learning_rate(args, optimizer_S, epoch)
        adjust_learning_rate_G(args, optimizer_G, epoch)
    global cur_batch_win
    if epoch > 0:
        net.train()
    loss_list, batch_list = [], []

    #for i, (images, labels) in enumerate(data_train_loader):
    for i in range(200): # 256 batch size * 200: 51200 images
        #images, labels = Variable(images).cuda(), Variable(labels).cuda()
        
        z = Variable(torch.randn(args.batch_size, args.latent_dim)).cuda()

        # optimizer.zero_grad()
        optimizer_G.zero_grad()
        optimizer_S.zero_grad()

        gen_imgs = generator(z)
        
        ### one-hot loss
        outputs_T, features_T = teacher(gen_imgs, out_feature=True)
        pred = outputs_T.data.max(1)[1]
        loss_activation = -features_T.abs().mean()
        loss_one_hot = criterion(outputs_T,pred)
        softmax_o_T = torch.nn.functional.softmax(outputs_T, dim = 1).mean(dim = 0)
        loss_information_entropy = (softmax_o_T * torch.log10(softmax_o_T)).sum()
        # loss = loss_one_hot * args.oh + loss_information_entropy * args.ie + loss_activation * args.a 

        loss = loss_one_hot
         
        ### KD loss
        outputs_S, features_S = net(gen_imgs, out_feature=True)
        loss_kd = kdloss(outputs_S, outputs_T)

        # ### only train student after n_epochs_G epochs
        # if epoch > args.n_epochs_G:
        #     loss = loss_kd
 
        ### from deepinversion: variation loss
        ## apply random jitter offsets
        off1 = random.randint(-lim_0, lim_0)
        off2 = random.randint(-lim_1, lim_1)
        gen_imgs = torch.roll(gen_imgs, shifts=(off1,off2), dims=(2,3))

        ## apply total variation regularization
        diff1 = gen_imgs[:,:,:,:-1] - gen_imgs[:,:,:,1:]
        diff2 = gen_imgs[:,:,:-1,:] - gen_imgs[:,:,1:,:]
        diff3 = gen_imgs[:,:,1:,:-1] - gen_imgs[:,:,:-1,1:]
        diff4 = gen_imgs[:,:,:-1,:-1] - gen_imgs[:,:,1:,1:]
        loss_var = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)

        ### R_feature loss
        loss_distr = sum([mod.r_feature for mod in loss_r_feature_layers])

        ### only train generator before n_epochs_G epoch
        if epoch <= args.n_epochs_G:
            loss = loss + (6e-3 * loss_var)
            loss = loss + (1.5e-5 * torch.norm(gen_imgs, 2))  # l2 loss
            loss = loss + 10*loss_distr                       # best for noise before BN
        ### only train student after n_epochs_G epochs
        elif epoch > args.n_epochs_G:
            if args.fix_G:
                loss = loss_kd
            else:
                loss = loss + (6e-3 * loss_var)
                loss = loss + (1.5e-5 * torch.norm(gen_imgs, 2))  # l2 loss
                if args.dataset == 'cifar10':
                    loss = loss + 10*loss_distr                       # best for noise before BN
                if args.dataset == 'cifar100':
                    loss = loss + loss_distr                       # best for noise before BN
                loss = loss + loss_kd

        loss_list.append(loss.data.item())
        batch_list.append(i+1)
 
        if i == 1:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.data.item()))
        if has_wandb:
            wandb.log({"loss/OneHot_Loss": loss_one_hot.item()})
            wandb.log({"loss/KD_Loss": loss_kd.item()})
            wandb.log({"loss/Var_Loss": loss_var.item()})
            wandb.log({"loss/R_Loss": loss_distr.item()})
            wandb.log({"loss/L2_Loss": torch.norm(gen_imgs, 2).item()})
            wandb.log({"total_loss": loss.data.item()})

        loss.backward()

        # optimizer.step()
        if epoch <= args.n_epochs_G:
            optimizer_G.step()
        if epoch > args.n_epochs_G:
            optimizer_S.step()
 
    # -------------------------------------------------
    print("-------> Model saved!!")

    save_name = "{}_bz{}_{}_ld{}_eN{}_eG{}_lrG{}_lrS{}.pth".format(args.dataset, 
                                                                    args.batch_size, args.fix_G, args.latent_dim,
                                                                    args.n_epochs, args.n_epochs_G,
                                                                    args.lr_G, args.lr_S)
    torch.save({'epoch': epoch,
                'S_state_dict': net.state_dict(),
                'S_optimizer_state_dict': optimizer_S.state_dict(),
                'G_state_dict': generator.state_dict(),
                'G_optimizer_state_dict':optimizer_G.state_dict()}, 
                save_path+"/"+save_name)
    
    # old_file = "bz{}_{}_ld{}_eN{}_eG{}_lrG{}_lrS{}_dcy{}_E{}.pth".format(args.batch_size, args.fix_G, args.latent_dim,
    #                                                                      args.n_epochs, args.n_epochs_G,
    #                                                                      args.lr_G, args.lr_S, args.decay,
    #                                                                      epoch-1)
    # if os.path.exists(save_path+"/"+old_file):
    #     os.remove(save_path+"/"+old_file)

    # -------------------------------------------------

# ------------------------------------

def test():
    global acc, acc_best
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            output = net(images)
            avg_loss += criterion(output, labels).sum()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
 
    avg_loss /= len(data_test)
    acc = float(total_correct) / len(data_test)
    if acc_best < acc:
        acc_best = acc

    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), acc))
    if has_wandb:
        wandb.log({"test_loss": avg_loss.data.item()})
        wandb.log({"test_acc": acc})


#############################################################

def train_and_test(epoch, args):
    train(epoch, args)
    test()
 
 
def main():
    if args.dataset == 'MNIST':
        epoch = 10
    else:
        epoch = args.n_epochs
    for e in range(start_epoch, epoch):
        train_and_test(e, args)
    # torch.save(net,args.output_dir + 'teacher'+args.ext)
 
#############################################################
if __name__ == '__main__':
    main()
