#Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import os, random
from DAFL_change.model.lenet import LeNet5
import DAFL_change.resnet as resnet
import torch
from torch.autograd import Variable
from torchvision.datasets.mnist import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 
import argparse

from script import *

import torch.nn as nn
import torch.nn.functional as F

try:
    import wandb
    has_wandb = True
except ImportError: 
    has_wandb = False

###########################################

parser = argparse.ArgumentParser(description='train-teacher-network')

# Basic model parameters.
parser.add_argument('--dataset', type=str, default='cifar10', choices=['MNIST','cifar10','cifar100'])
parser.add_argument('--data', type=str, default='cache/data/')
parser.add_argument('--output_dir', type=str, default='cache/models/')
parser.add_argument('--teacher_dir', type=str, default='cache/models/')
parser.add_argument('--ext', type=str, default='')
parser.add_argument('--resume', action='store_true', default=False,
                    help='whether to resume from ckpt')

parser.add_argument('--train_G', action='store_true', default=False,
                    help='whether to train multiple generators')
parser.add_argument('--train_S', action='store_true', default=False,
                    help='whether to train student')

parser.add_argument('--n_divid', type=int, default=10, help='number of division of dataset')

parser.add_argument('--n_epochs_G', type=int, default=50, help='number of epochs of training generator')
parser.add_argument('--n_epochs', type=int, default=400, help='number of epochs of training total')

parser.add_argument('--fix_G', action='store_true', default=False,
                    help='whether stop train generator after start training student')

parser.add_argument('--batch_size', type=int, default=256, help='size of the batches')
parser.add_argument('--lr_G', type=float, default=0.001, help='learning rate of generator')
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
    if args.train_G:
        id = "trainG-{}-bz{}-{}-ld{}-eN{}-eG{}-lrG{}-lrS{}".format(args.ext, args.batch_size, args.fix_G, args.latent_dim,
                                                                        args.n_epochs, args.n_epochs_G,
                                                                        args.lr_G, args.lr_S)
    if args.train_S:
        id = "trainS-{}-bz{}-{}-ld{}-eN{}-eG{}-lrG{}-lrS{}".format(args.ext, args.batch_size, args.fix_G, args.latent_dim,
                                                                        args.n_epochs, args.n_epochs_G,
                                                                        args.lr_G, args.lr_S)
    wandb.init(project='few-shot-multi', entity='zhoushanglin100', config=args, resume="allow", id=id)
    # wandb.init(project='few-shot-multi', entity='zhoushanglin100', config=args)
    wandb.config.update(args)

acc = 0
acc_best = 0

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

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if has_wandb:
        wandb.log({"lr/lr_G": lr})

# ---------------------------------------------

def kdloss(y, teacher_scores):
    p = F.log_softmax(y, dim=1)
    q = F.softmax(teacher_scores, dim=1)
    l_kl = F.kl_div(p, q, size_average=False)  / y.shape[0]
    return l_kl

######################################################

def train_G(args, idx, net, generator, teacher, epoch,
            optimizer_S, optimizer_G, criterion, 
            loss_r_feature_layers, lim_0, lim_1,
            mean, var):

    if args.dataset != 'MNIST':
        # adjust_learning_rate(args, optimizer_S, epoch)
        adjust_learning_rate_G(args, optimizer_G, epoch)
    
    if epoch > 0:
        net.train()
    
    for i in range(200):

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

        ### from deepinversion: variation loss
        ## apply random jitter offsets
        off1 = random.randint(-lim_0, lim_0)
        off2 = random.randint(-lim_1, lim_1)
        gen_imgs = torch.roll(gen_imgs, shifts=(off1, off2), dims=(2,3))

        ## apply total variation regularization
        diff1 = gen_imgs[:,:,:,:-1] - gen_imgs[:,:,:,1:]
        diff2 = gen_imgs[:,:,:-1,:] - gen_imgs[:,:,1:,:]
        diff3 = gen_imgs[:,:,1:,:-1] - gen_imgs[:,:,:-1,1:]
        diff4 = gen_imgs[:,:,:-1,:-1] - gen_imgs[:,:,1:,1:]
        loss_var = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)

        ### R_feature loss (ToDo)
        # loss_distr = sum([mod.r_feature for mod in loss_r_feature_layers])
        # nch_G = gen_imgs[0].shape[1]
        # mean_G = gen_imgs[0].mean([0, 2, 3])
        # var_G = gen_imgs[0].permute(1, 0, 2, 3).contiguous().view([nch_G, -1]).var(1, unbiased=False)

        nch_G = gen_imgs.shape[1]
        mean_G = gen_imgs.mean([0, 2, 3])
        var_G = gen_imgs.permute(1, 0, 2, 3).contiguous().view([nch_G, -1]).var(1, unbiased=False)

        loss_distr = torch.norm(var_G - var, 2) + torch.norm(mean_G - mean, 2)

        ### only train generator before n_epochs_G epoch
        loss = loss + (6e-3 * loss_var)
        loss = loss + (1.5e-5 * torch.norm(gen_imgs, 2))  # l2 loss
        loss = loss + 10*loss_distr                       # best for noise before BN

        if i % 10 == 0:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.data.item()))

        if has_wandb:
            wandb.log({"loss_G/OneHot_Loss_"+str(idx): loss_one_hot.item()})
            wandb.log({"loss_G/KD_Loss_S_"+str(idx): loss_kd.item()})
            wandb.log({"loss_G/Var_Loss_"+str(idx): loss_var.item()})
            wandb.log({"loss_G/R_Loss_"+str(idx): loss_distr.item()})
            wandb.log({"loss_G/L2_Loss_"+str(idx): torch.norm(gen_imgs, 2).item()})
            wandb.log({"G_perf/total_loss_"+str(idx): loss.data.item()})

        loss.backward()
        optimizer_G.step()


# --------------------

def train_S(args, net, G_list, teacher, epoch, optimizer_S, losses):
    
    if args.dataset != 'MNIST':
        adjust_learning_rate(args, optimizer_S, epoch)
    
    if epoch > 0:
        net.train()

    for i in range(200):

        loss_total = None

        optimizer_S.zero_grad()
        
        for tidx, generator in enumerate(G_list):

            ### KD loss
            
            z = Variable(torch.randn(args.batch_size, args.latent_dim)).cuda()
            gen_imgs = generator(z)

            outputs_T, features_T = teacher(gen_imgs, out_feature=True)
            outputs_S, features_S = net(gen_imgs, out_feature=True)
            loss_kd = kdloss(outputs_S, outputs_T)

            ### only train student after n_epochs_G epochs
            if args.fix_G:
                loss = loss_kd
            else:
                print("To-Do")
                # loss = loss + (6e-3 * loss_var)
                # loss = loss + (1.5e-5 * torch.norm(gen_imgs, 2))      # l2 loss
                # if args.dataset == 'cifar10':
                #     loss = loss + 10*loss_distr                       # best for noise before BN
                # if args.dataset == 'cifar100':
                #     loss = loss + loss_distr                          # best for noise before BN
                # loss = loss + loss_kd

            if loss_total is None:
                loss_total = loss
            else:
                loss_total += loss
            losses[tidx].append(loss.item())

            # print('Train - generator %d, Loss: %f' % (tidx, loss_total.data.item()))
            
            if has_wandb:
                wandb.log({"loss_S/KD_Loss_G_"+str(tidx): loss.item()})
                # wandb.log({"loss_S/KD_Loss_S": loss.item()})
                wandb.log({"total_loss_S": loss_total.item()})

        if i % 10 == 0:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss_total.data.item()))

        loss_total.backward()
        optimizer_S.step()

# ------------------------------------

def test(args, net, data_test_loader, criterion):

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
 
    avg_loss /= len(data_test_loader)
    acc = float(total_correct) / len(data_test_loader)

    if acc_best < acc:
        acc_best = acc
        
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), acc))
    
    if has_wandb:
        wandb.log({"test_loss": avg_loss.data.item()})
        wandb.log({"test_acc": acc})


# ------------------------------------

def test_S(args, net, len_G, num_classes, criterion):

    net.eval()

    losses = [0 for _ in range(len_G)]
    correct = [0 for _ in range(len_G)]
    total = [0 for _ in range(len_G)]
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():

        batch_idx = 0
        
        for i in range(len_G):
            _, test_loader = get_split_cifar10(args, args.batch_size, num_classes*i, num_classes*(i+1))

            for images, labels in test_loader:
                images, labels = Variable(images).cuda(), Variable(labels).cuda()
                outputs = net(images)
                loss = criterion(outputs, labels)

                losses[i] += loss.item()
                _, predicted = outputs.topk(5, 1, True, True)
                predicted = predicted.t()
                correct[i] += predicted.eq(labels.view(1, -1).expand_as(predicted)).sum().item()

                total[i] += labels.size(0)
                batch_idx += 1

            print('Generator {}:'.format(i + 1))
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (losses[i] / (1 * (batch_idx + 1)), 100. * correct[i] / total[i], correct[i], total[i]))

            loss_tmp = losses[i] / (1 * (batch_idx + 1))
            acc_tmp = 100. * correct[i] / total[i]
            if has_wandb:
                wandb.log({"S_perf_test/S_test_loss_"+str(i+1): loss_tmp})
                wandb.log({"S_perf_test/S_test_acc_"+str(i+1): acc_tmp})

    acc_total = 100. * sum(correct) / sum(total)
    if has_wandb:
        wandb.log({"S_test_acc_total": acc_total})

    print('>>>>> Total:')
    print('Acc: %.3f%% (%d/%d)' % (100. * sum(correct) / sum(total), sum(correct), sum(total)))
    print('>>>>> Finished student validation')

#############################################################

def main():

    global acc, acc_best
    os.makedirs(args.output_dir, exist_ok=True)  

    acc = 0
    acc_best = 0
    start_epoch = 1

    # -----------------------------------------------

    teacher = torch.load(args.teacher_dir + 'teacher_acc_95.3').cuda()
    teacher.eval()
    teacher = nn.DataParallel(teacher)

    # -------------------------------------
    save_path = 'cache/ckpts/multi_'+args.ext
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # ------------------------------------------------

    ## Create hooks for feature statistics catching
    loss_r_feature_layers = []
    for module in teacher.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(DeepInversionFeatureHook(module))
   
    # setting up the range for jitter
    lim_0, lim_1 = 2, 2

    # ------------------------------------------------
    ### train generator
    # if start_epoch <= args.n_epochs_G:
    if args.train_G:

        n = int(args.n_divid)

        ### specific for cifar10
        num_classes = int(10/n)

        for idx in range(0, n):
            start_class = idx*num_classes
            end_class = (idx+1)*num_classes

            print("\n !!!!! start_class: "+str(start_class)+" end_class: "+str(end_class))
            # ------------------------------------------------

            generator = Generator().cuda()
            generator = nn.DataParallel(generator)

            # ------------------------------------------------
            ### Create dataset
                
            if args.dataset == 'cifar10':
                data_train_loader, data_test_loader = get_split_cifar10(args, args.batch_size, start_class, end_class)
                # print(len(data_train_loader), len(data_test_loader))
                criterion = torch.nn.CrossEntropyLoss().cuda()
                optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr_G)

                net = resnet.ResNet18().cuda()
                criterion = torch.nn.CrossEntropyLoss().cuda()
                optimizer_S = torch.optim.SGD(net.parameters(), lr=args.lr_S, momentum=0.9, weight_decay=5e-4)

                train_sample_loader, test_sample_loader = get_split_cifar10(args, 100, start_class, end_class)

            for i, (images, labels) in enumerate(train_sample_loader):
                if i == 0:
                    images, labels = images.cuda(), labels.cuda()

                    # print("!!!!!!!!!!!!!")
                    # print(images.shape, labels.shape)

                    nch = images.shape[1]
                    mean = images.mean([0, 2, 3])
                    var = images.permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

                    print(mean, var)

                    break

            # ------------------------------------------------
            ### start training generator

            for e in range(start_epoch, args.n_epochs_G):
                if has_wandb:
                    wandb.log({"epoch": e})

            # for e in range(start_epoch, 2):

                train_G(args, idx, net, generator, teacher, e, 
                            optimizer_S, optimizer_G, criterion, 
                            loss_r_feature_layers, lim_0, lim_1,
                            mean, var)

                test(args, net, data_test_loader, criterion)
            
                save_name = "start-"+str(start_class)+"_end-"+str(end_class)+".pth"
                torch.save({'epoch': e,
                            'G_state_dict': generator.state_dict(),
                            'G_optimizer_state_dict':optimizer_G.state_dict()}, 
                            save_path+"/"+save_name)

    # ------------------------------------------------
    ### train student
    # if start_class > args.n_epochs_G:
    if args.train_S:

        start_epoch = args.n_epochs_G

        G_list = []
        num_G = int(args.n_divid)
        num_classes = int(10/num_G)            ### specific for cifar10

        for i in range(0, num_G):
        # for i in range(0, 3):

            start_class = i*num_classes
            end_class = (i+1)*num_classes

            generator = Generator().cuda()
            generator = nn.DataParallel(generator)

            # ------------------------------------------------
            G_name = "start-"+str(start_class)+"_end-"+str(end_class)+".pth"
            print(save_path+'/'+G_name)
            ckeckpoints = torch.load(save_path+'/'+G_name)
            generator.load_state_dict(ckeckpoints['G_state_dict'])
            generator = generator.eval()
            G_list.append(generator)

        # ------------------------------------------------
        if args.dataset == 'cifar10':
            data_train_loader, data_test_loader = get_split_cifar10(args, args.batch_size, start_class, end_class)
            net = resnet.ResNet18().cuda()
            criterion = torch.nn.CrossEntropyLoss().cuda()
            optimizer_S = torch.optim.SGD(net.parameters(), lr=args.lr_S, momentum=0.9, weight_decay=5e-4)

        # ------------------------------------------------

        if args.resume:
            load_name = "trainS_{}_ld{}_eN{}_eG{}_lrG{}_lrS{}.pth".format(args.fix_G, args.latent_dim,
                                                                            args.n_epochs, args.n_epochs_G,
                                                                            args.lr_G, args.lr_S)
            # load_name = 'student.pth'
            if os.path.exists(save_path+'/'+load_name):
                checkpoint = torch.load(save_path+'/'+load_name)
                net.load_state_dict(checkpoint['S_state_dict'])
                optimizer_S.load_state_dict(checkpoint['S_optimizer_state_dict'])
                resume_epoch = checkpoint['epoch']
                start_epoch = resume_epoch+1
        # ------------------------------------------------

        for e in range(start_epoch, args.n_epochs):

            if has_wandb:
                wandb.log({"epoch": e})

            losses = [[] for _ in range(len(G_list))]
            accuracy = [[] for _ in range(len(G_list))]

            train_S(args, net, G_list, teacher, e, optimizer_S, losses)

            if e % 1 == 0:
                print(">>> Checking student accuracy")
                test_S(args, net, len(G_list), num_classes, criterion)

            test(args, net, data_test_loader, criterion)

            #### save student model
            print("-------> Model saved!!")
            save_name = "trainS_{}_ld{}_eN{}_eG{}_lrG{}_lrS{}.pth".format(args.fix_G, args.latent_dim,
                                                                            args.n_epochs, args.n_epochs_G,
                                                                            args.lr_G, args.lr_S)
            # save_name = "student.pth"
            torch.save({'epoch': e,
                        'S_state_dict': net.state_dict(),
                        'S_optimizer_state_dict': optimizer_S.state_dict()},
                        save_path+"/"+save_name)

#############################################################
if __name__ == '__main__':
    main()
