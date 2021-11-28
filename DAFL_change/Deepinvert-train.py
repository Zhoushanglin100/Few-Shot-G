#Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import os
from lenet import LeNet5
import resnet
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

# from deep invert
import random
import torch.backends.cudnn as cudnn
import torch.optim as optim
# import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils

import numpy as np
import os
import glob
import collections


parser = argparse.ArgumentParser(description='train-teacher-network')

# Basic model parameters.
parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST','cifar10','cifar100'])
parser.add_argument('--data', type=str, default='cache/data/')
parser.add_argument('--output_dir', type=str, default='cache/models/')
parser.add_argument('--teacher_dir', type=str, default='cache/models/')
# from deep invert
parser.add_argument('--bs', default=256, type=int, help='batch size')
parser.add_argument('--iters_mi', default=2000, type=int, help='number of iterations for model inversion')
parser.add_argument('--cig_scale', default=0.0, type=float, help='competition score')
parser.add_argument('--di_lr', default=0.1, type=float, help='lr for deep inversion')
parser.add_argument('--di_var_scale', default=2.5e-5, type=float, help='TV L2 regularization coefficient')
parser.add_argument('--di_l2_scale', default=0.0, type=float, help='L2 regularization coefficient')
parser.add_argument('--r_feature_weight', default=1e2, type=float, help='weight for BN regularization statistic')
parser.add_argument('--amp', action='store_true', help='use APEX AMP O1 acceleration')


args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)  

acc = 0
acc_best = 0

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

    data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
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

    data_train = CIFAR10(args.data,
                       transform=transform_train)
    data_test = CIFAR10(args.data,
                      train=False,
                      transform=transform_test)

    data_train_loader = DataLoader(data_train, batch_size=128, shuffle=True, num_workers=8)
    data_test_loader = DataLoader(data_test, batch_size=100, num_workers=0)

    #net = resnet.ResNet34().cuda()
    net = resnet.ResNet18().cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

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
                      
    data_train_loader = DataLoader(data_train, batch_size=128, shuffle=True, num_workers=0)
    data_test_loader = DataLoader(data_test, batch_size=128, num_workers=0)
    net = resnet.ResNet34(num_classes=100).cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)


def adjust_learning_rate(optimizer, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    if epoch < 80:
        lr = 0.1
    elif epoch < 120:
        lr = 0.01
    else:
        lr = 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

teacher = torch.load(args.teacher_dir + 'teacher_acc_95.3').cuda()
teacher.eval()
teacher = nn.DataParallel(teacher)

#deep invert
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

        # forcing mean and variance to match between two distributions
        # other ways might work better, e.g. KL divergence
        r_feature = torch.norm(module.running_var.data.type(var.type()) - var, 2) + torch.norm(
            module.running_mean.data.type(var.type()) - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()

def get_images(net, bs=256, epochs=2000, idx=-1, var_scale=0.00005,
               net_student=None, prefix=None, competitive_scale=0.01, train_writer = None, global_iteration=None,
               use_amp=False,
               optimizer = None, inputs = None, bn_reg_scale = 0.0, random_labels = False, l2_coeff=0.0):
    kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()
    best_cost = 1e6
    
    # initialize gaussian inputs
    inputs.data = torch.randn((bs, 3, 32, 32), requires_grad=True, device='cuda')
    # set up criteria for optimization
    criterion = nn.CrossEntropyLoss()
    optimizer.state = collections.defaultdict(dict)  # Reset state of optimizer
    # target outputs to generate
    if random_labels:
        targets = torch.LongTensor([random.randint(0,9) for _ in range(bs)]).to('cuda')
    else:
        targets = torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 25 + [0, 1, 2, 3, 4, 5]).to('cuda')

    ## Create hooks for feature statistics catching
    loss_r_feature_layers = []
    for module in net.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(DeepInversionFeatureHook(module))

    # setting up the range for jitter
    lim_0, lim_1 = 2, 2
    for epoch in range(epochs):
        # apply random jitter offsets
        off1 = random.randint(-lim_0, lim_0)
        off2 = random.randint(-lim_1, lim_1)
        inputs_jit = torch.roll(inputs, shifts=(off1,off2), dims=(2,3))

        # foward with jit images
        optimizer.zero_grad()
        net.zero_grad()
        outputs = net(inputs_jit)
        loss = criterion(outputs, targets)
        loss_target = loss.item()


        # apply total variation regularization
        diff1 = inputs_jit[:,:,:,:-1] - inputs_jit[:,:,:,1:]
        diff2 = inputs_jit[:,:,:-1,:] - inputs_jit[:,:,1:,:]
        diff3 = inputs_jit[:,:,1:,:-1] - inputs_jit[:,:,:-1,1:]
        diff4 = inputs_jit[:,:,:-1,:-1] - inputs_jit[:,:,1:,1:]
        loss_var = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
        loss = loss + var_scale*loss_var

        # R_feature loss
        loss_distr = sum([mod.r_feature for mod in loss_r_feature_layers])
        loss = loss + bn_reg_scale*loss_distr # best for noise before BN

        # l2 loss
        if 1:
            loss = loss + l2_coeff * torch.norm(inputs_jit, 2)

        if epoch % 100==0:
            print(f"It {epoch}\t Losses: total: {loss.item():3.3f},\ttarget: {loss_target:3.3f} \tR_feature_loss unscaled:\t {loss_distr.item():3.3f}")

        if best_cost > loss.item():
            best_cost = loss.item()
            best_inputs = inputs.data

        loss.backward()

        optimizer.step()
    return best_inputs

    
def kdloss(y, teacher_scores):
    p = F.log_softmax(y, dim=1)
    q = F.softmax(teacher_scores, dim=1)
    l_kl = F.kl_div(p, q, size_average=False)  / y.shape[0]
    return l_kl
        
def train(epoch, optimizer_di, inputs):
    if args.dataset != 'MNIST':
        adjust_learning_rate(optimizer, epoch)
    global cur_batch_win
    #if args.amp:
    #    # need to do this trick for FP16 support of batchnorms
    #    net.train()
    #    for module in net.modules():
    #        if isinstance(module, nn.BatchNorm2d):
    #            module.eval().half()
    net.train()
    loss_list, batch_list = [], []
    


    # training and using the images from deepinvert
    # for i, (images, labels) in enumerate(data_train_loader):
    for i in range(200):
        #images, labels = Variable(images).cuda(), Variable(labels).cuda()
        images = get_images(net=teacher, bs=args.bs, epochs=args.iters_mi,
                        optimizer=optimizer_di, inputs=inputs, bn_reg_scale=args.r_feature_weight,
                        var_scale=args.di_var_scale, random_labels=False, l2_coeff=args.di_l2_scale)
        optimizer.zero_grad()
 
        output = net(images)
 
        # loss = criterion(output, labels)
        # kd
        outputs_T, features_T = teacher(images, out_feature=True)
        outputs_S, features_S = net(images, out_feature=True)
        loss_kd = kdloss(outputs_S, outputs_T)
        loss = loss_kd


        loss_list.append(loss.data.item())
        batch_list.append(i+1)
 
        if i == 1:
            #print("loss_kd:")
            #print(loss_kd)
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.data.item()))
 
        loss.backward()
        optimizer.step()
        test() 
 
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
 
 
def train_and_test(epoch, optimizer_di, inputs):
    train(epoch, optimizer_di, inputs)
    test()
 
 
def main():
    # deep invert
    data_type = torch.half if args.amp else torch.float
    inputs = torch.randn((args.bs, 3, 32, 32), requires_grad=True, device='cuda', dtype=data_type)

    optimizer_di = optim.Adam([inputs], lr=args.di_lr)

    if args.amp:
        opt_level = "O1"
        loss_scale = 'dynamic'

        [teacher], optimizer_di = amp.initialize(
            [teacher], optimizer_di,
            opt_level=opt_level,
            loss_scale=loss_scale)
    if args.amp:
        # need to do this trick for FP16 support of batchnorms
        teacher.train()
        for module in teacher.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval().half()
    if args.dataset == 'MNIST':
        epoch = 10
    else:
        epoch = 200
    for e in range(1, epoch):
        train_and_test(e, optimizer_di, inputs)
    torch.save(net,args.output_dir + 'teacher')
 
 
if __name__ == '__main__':
    main()
