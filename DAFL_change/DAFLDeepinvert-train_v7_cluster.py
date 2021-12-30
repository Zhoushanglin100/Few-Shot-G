import os, random
import resnet as resnet
import torch
from torch.autograd import Variable
import argparse

from script import *
import collections

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

try:
    import wandb
    has_wandb = True
except ImportError: 
    has_wandb = False

# has_wandb = False

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

parser.add_argument('--n_divid', type=int, default=5, help='number of division of dataset')
parser.add_argument('--total_class', type=int, default=10, help='total number of classes of dataset')

parser.add_argument('--n_epochs_G', type=int, default=50, help='number of epochs of training generator')
parser.add_argument('--n_epochs', type=int, default=2000, help='number of epochs of training total')

parser.add_argument('--fix_G', action='store_true', default=False,
                    help='whether stop train generator after start training student')

parser.add_argument('--hook_type', type=str, default='output', choices=['input', 'output'],
                    help = "hook statistics from input data or output data")
parser.add_argument('--stat_type', type=str, default='extract', choices=['running', 'extract'],
                    help = "statistics from self extracted from a batch or saved stats from teacher")

parser.add_argument('--batch_size', type=int, default=128, help='size of the batches')
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

print("-----------------------------")
print(args)
print("-----------------------------")

if has_wandb:
    if args.train_G:
        # id = "trainG-{}-bz{}-{}-ld{}-eN{}-eG{}-lrG{}-lrS{}".format(args.ext, 
        #                                                            args.batch_size, 
        #                                                            args.fix_G, 
        #                                                            args.latent_dim,
        #                                                            args.n_epochs, args.n_epochs_G,
        #                                                            args.lr_G, args.lr_S)
        id = "{}_trainG-{}".format(args.dataset, args.ext)
    if args.train_S:
        # id = "trainS-{}-bz{}-{}-ld{}-eN{}-eG{}-lrG{}-lrS{}".format(args.ext, 
        #                                                             args.batch_size, 
        #                                                             args.fix_G, 
        #                                                             args.latent_dim,
        #                                                             args.n_epochs, args.n_epochs_G,
        #                                                             args.lr_G, args.lr_S)
        id = "{}_trainS-{}".format(args.dataset, args.ext)
    wandb.init(project='few-shot-multi', entity='tidedancer', config=args, resume="allow", id=id)
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

    def __init__(self, args, name, module, mean_dict, var_dict):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.name = name
        self.mean_dict = mean_dict
        self.var_dict = var_dict
        self.hook_type = args.hook_type
        self.stat_type = args.stat_type
        self.mean_layers = []

    def hook_fn(self, module, input, output):

        # hook co compute deepinversion's feature distribution regularization
        if self.hook_type == "input":
            nch = input[0].shape[1]
            mean = input[0].mean([0, 2, 3])
            var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        elif self.hook_type == "output":
            nch = output.shape[1]
            mean = output.mean([0,2,3])
            var = output.permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

            self.mean_layers.append(mean)

        if self.stat_type == "running":
            # forcing mean and variance to match between two distributions
            # other ways might work better, e.g. KL divergence
            r_feature = torch.norm(module.running_var.data.type(var.type()) - var, 2) + torch.norm(
                module.running_mean.data.type(mean.type()) - mean, 2)
        elif self.stat_type == "extract":
            batch_mean = self.mean_dict[self.name].cuda()
            batch_var = self.var_dict[self.name].cuda()
            
            # print("!!!!!!!!!", self.name)
            # print("output", mean.shape, var.shape)
            # print("batch", batch_mean.shape, batch_var.shape)
            # print("=====")

            criterion = nn.CosineEmbeddingLoss()
            r_feature = criterion(batch_var.view(1,-1),var.view(1,-1),torch.ones(1).cuda())+criterion(batch_mean.view(1,-1),mean.view(1,-1),torch.ones(1).cuda())
            
            # r_feature = torch.norm(module.running_var.data.type(var.type()) - var, 2) + torch.norm(
            #     module.running_mean.data.type(mean.type()) - mean, 2)

            # r_feature = torch.norm(batch_var.type(var.type()) - var, 2) + torch.norm(
            #         batch_mean.type(mean.type()) - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()


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
    
    print("!!!!", lr)

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
            lim_0, lim_1, # mean, var,
            loss_r_feature_layers): 

    print("\n>>>>> Train Generators <<<<<\n")

    if args.dataset != 'MNIST':
        adjust_learning_rate_G(args, optimizer_G, epoch)
    
    net.train()
    loss = None

    for i in range(200):
    # for i in range(3):

        z = Variable(torch.randn(args.batch_size, args.latent_dim)).cuda()

        optimizer_G.zero_grad()

        gen_imgs = generator(z)

        ### one-hot loss
        outputs_T, features_T = teacher(gen_imgs, out_feature=True)

        pred = outputs_T.data.max(1)[1]
        loss_activation = -features_T.abs().mean()
        loss_one_hot = criterion(outputs_T,pred)
        softmax_o_T = torch.nn.functional.softmax(outputs_T, dim = 1).mean(dim = 0)
        loss_information_entropy = (softmax_o_T * torch.log10(softmax_o_T)).sum()
        # loss = loss_one_hot * args.oh + loss_information_entropy * args.ie + loss_activation * args.a 
        
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
        loss_distr = sum([mod.r_feature for mod in loss_r_feature_layers])

        # nch_G = gen_imgs.shape[1]
        # mean_G = gen_imgs.mean([0, 2, 3])
        # var_G = gen_imgs.permute(1, 0, 2, 3).contiguous().view([nch_G, -1]).var(1, unbiased=False)
        # loss_distr = torch.norm(var_G - var, 2) + torch.norm(mean_G - mean, 2)

        ### only train generator before n_epochs_G epoch
        loss = loss_one_hot
        loss += (6e-3 * loss_var)
        loss += (1.5e-5 * torch.norm(gen_imgs, 2))  # l2 loss
        # loss += 10*loss_distr                       # best for noise before BN
        loss += 100*loss_distr                       # best for noise before BN

        if i % 10 == 0:
            print('Train G_%d, Epoch %d, Batch: %d, Loss: %f' % (idx, epoch, i, loss.data.item()))

        if has_wandb:
            wandb.log({"loss_G/OneHot_Loss_"+str(idx): loss_one_hot.item()})
            wandb.log({"loss_G/KD_Loss_S_"+str(idx): loss_kd.item()})
            wandb.log({"loss_G/Var_Loss_"+str(idx): loss_var.item()})
            wandb.log({"loss_G/R_Loss_"+str(idx): loss_distr.item()})
            wandb.log({"loss_G/L2_Loss_"+str(idx): torch.norm(gen_imgs, 2).item()})
            wandb.log({"G_perf/total_loss_"+str(idx): loss.data.item()})

        # loss.backward(retain_graph=True)
        loss.backward()
        optimizer_G.step()


# --------------------

def train_S(args, net, G_list, teacher, epoch, optimizer_S):
    print(">>>>> Train Student <<<<<")

    net.train()

    for i in range(200):
    # for i in range(5):

        loss_total = None

        optimizer_S.zero_grad()

        gen_imgs = Variable(torch.randn(args.batch_size*10, args.latent_dim)).cuda()

        for gidx, generator in enumerate(G_list):

            generator = generator.cuda()

            ### KD loss
            z = Variable(torch.randn(args.batch_size, args.latent_dim)).cuda()
            gen_imgs_gid = generator(z)
            # print("------>", gen_imgs_gid.shape)

            if gidx == 0:
                gen_imgs = gen_imgs_gid
            else:
                gen_imgs = torch.cat((gen_imgs, gen_imgs_gid), 0)

        # print("======>>>", i, gen_imgs.shape)
        # exit(0)

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

        if has_wandb:
            wandb.log({"total_loss_S": loss.item()})

        if i % 10 == 0:
            print('Student Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.data.item()))

        loss.backward()
        optimizer_S.step()

# ------------------------------------

def test(args, net, data_test_loader, criterion):

    # global acc, acc_best

    net.eval()
    total_correct = 0
    avg_loss = 0.0
    total_len = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            output = net(images)
            # print(output)
            avg_loss += criterion(output, labels).sum()
            pred = output.data.max(1)[1]
            # print(pred)
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
            total_len += labels.size(0)

    avg_loss /= len(data_test_loader)
    acc = float(total_correct) / total_len

    # if acc_best < acc:
    #     acc_best = acc
        
    print('\n|||| Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), acc))
    
    if has_wandb:
        wandb.log({"test_loss": avg_loss.data.item()})
        wandb.log({"test_acc": acc})


# ------------------------------------

def test_S(args, net, len_G, num_classes, criterion):

    net.eval()

    losses = [0 for _ in range(len_G)]
    correct = [0 for _ in range(len_G)]
    total = [0 for _ in range(len_G)]

    with torch.no_grad():

        # batch_idx = 0
        
        for i in range(len_G):

            start_class = i*num_classes
            end_class = (i+1)*num_classes
            print("test_S start_class: "+str(start_class)+" end_class: "+str(end_class))

            if args.dataset == 'cifar10':
                _, test_loader = get_split_cifar10(args, args.batch_size, start_class, end_class*(i+1))
            elif args.dataset == 'cifar100':
                _, test_loader = get_split_cifar100(args, args.batch_size, start_class, end_class*(i+1))

            for images, labels in test_loader:
                images, labels = Variable(images).cuda(), Variable(labels).cuda()
                outputs = net(images)
                loss = criterion(outputs, labels).sum()

                losses[i] += loss.item()
                pred = outputs.data.max(1)[1]
                correct[i] += pred.eq(labels.data.view_as(pred)).sum().item()
                
                # losses[i] += loss.item()
                # _, predicted = outputs.topk(5, 1, True, True)
                # predicted = predicted.t()
                # correct[i] += predicted.eq(labels.view(1, -1).expand_as(predicted)).sum().item()
                total[i] += labels.size(0)
                # batch_idx += 1

            # print('Generator {}:'.format(i + 1))
            # print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (losses[i]/(1*(batch_idx+1)), 100.*correct[i]/total[i], correct[i], total[i]))

            loss_tmp = losses[i]/len(test_loader)
            acc_tmp = 100. * correct[i] / total[i]

            # loss_tmp = losses[i]/(1*(batch_idx+1))
            # acc_tmp = 100.*correct[i]/total[i]

            if has_wandb:
                wandb.log({"S_perf_test/S_test_loss_"+str(i+1): loss_tmp})
                wandb.log({"S_perf_test/S_test_acc_"+str(i+1): acc_tmp})

    acc_total = 100. * sum(correct) / sum(total)
    if has_wandb:
        wandb.log({"S_test_acc_total": acc_total})

    print('>>>>> Total:')
    print('Acc: %.3f%% (%d/%d)' % (100* sum(correct) / sum(total), sum(correct), sum(total)))

#############################################################

def main():

    global acc, acc_best
    os.makedirs(args.output_dir, exist_ok=True)  

    acc = 0
    acc_best = 0
    start_epoch = 1

    # -----------------------------------------------
    if args.dataset == "cifar10":
        teacher = torch.load(args.teacher_dir + 'teacher_acc_95.3').cuda()
    elif args.dataset == "cifar100":
        teacher = resnet.ResNet34(num_classes=100).cuda()
        ckpt_teacher = torch.load("cache/pretrained/cifar100_resnet34.pth")
        teacher.load_state_dict(ckpt_teacher['state_dict'])
    else:
        teacher = models.resnet34(pretrained=True)

    teacher.eval()
    teacher = nn.DataParallel(teacher)
    
    # print("||||||||||||||")
    # for name, module in teacher.named_modules():
    #     print(name)
    # print("||||||||||||||")
    # exit(0)

    # -------------------------------------
    save_path = 'cache/ckpts_'+args.dataset+'/multi_'+args.ext
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # ------------------------------------------------
    ### train generator
    # if start_epoch <= args.n_epochs_G:
    if args.train_G:

        # setting up the range for jitter
        lim_0, lim_1 = 2, 2

        n = int(args.n_divid)
        total = int(args.total_class)
        num_classes = int(total/n)

        ### iteratively train generators
        for idx in range(0, n):
            
            # start_class = idx*num_classes
            # end_class = (idx+1)*num_classes

            start_class = idx
            end_class = idx+1

            print("\n !!!!! start_class: "+str(start_class)+" end_class: "+str(end_class))
            # ------------------------------------------------

            generator = Generator().cuda()
            generator = nn.DataParallel(generator)

            # ------------------------------------------------
                
            if args.dataset == 'cifar10':

                ### Create dataset
                # data_train_loader, data_test_loader = get_split_cifar10(args, args.batch_size, start_class, end_class)

                ### optimization
                criterion = torch.nn.CrossEntropyLoss().cuda()
                optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr_G)
                ### student
                net = resnet.ResNet18().cuda()
                optimizer_S = torch.optim.SGD(net.parameters(), lr=args.lr_S, momentum=0.9, weight_decay=5e-4)
                
                ### Create hooks for feature statistics catching
                stat_path = "stats_"+args.dataset+"/stats_multi_"+args.ext+"/"+args.hook_type
                mean_layers_dictionary = torch.load(stat_path+"/mean_resnet34_start-"+str(start_class)+"_end-"+str(end_class)+".pth")
                var_layers_dictionary = torch.load(stat_path+"/var_resnet34_start-"+str(start_class)+"_end-"+str(end_class)+".pth")

            if args.dataset == 'cifar100':
                ### optimization
                criterion = torch.nn.CrossEntropyLoss().cuda()
                optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr_G)
                ### student
                net = resnet.ResNet18(num_classes=100).cuda()
                optimizer_S = torch.optim.SGD(net.parameters(), lr=args.lr_S, momentum=0.9, weight_decay=5e-4)
                
                ### Create hooks for feature statistics catching
                stat_path = "stats_"+args.dataset+"/stats_multi_"+args.ext+"/"+args.hook_type
                mean_layers_dictionary = torch.load(stat_path+"/mean_resnet34_start-"+str(start_class)+"_end-"+str(end_class)+".pth")
                var_layers_dictionary = torch.load(stat_path+"/var_resnet34_start-"+str(start_class)+"_end-"+str(end_class)+".pth")



            # print("\n||||||||||||||")
            # for name, W in teacher.named_parameters():
            #     if ('bn' in name) and ("weight" in name):
            #         print(name, W.shape)

            # # for i in mean_layers_dictionary:
            # #     print(i, mean_layers_dictionary[i].shape, var_layers_dictionary[i].shape)

            # # print("mean_layers_dictionary", mean_layers_dictionary.keys())
            # # print("\nvar_layers_dictionary", var_layers_dictionary.keys())
            # print("||||||||||||||\n")
            # exit(0)

            mean_layers_dictionary = {f'module.{k}': v for k, v in mean_layers_dictionary.items()}
            var_layers_dictionary = {f'module.{k}': v for k, v in var_layers_dictionary.items()}

            loss_r_feature_layers = []
            name_layer = []
            for name, module in teacher.named_modules():
                # if isinstance(module, nn.BatchNorm2d):
                if ("bn" in name) or ("downsample.1" in name):
                # if name in mean_layers_dictionary.keys():
                    # print("1111", name)
                    aa = DeepInversionFeatureHook(args, name, module, mean_layers_dictionary, var_layers_dictionary)
                    loss_r_feature_layers.append(aa)
                    name_layer.append(name)


            # ------------------------------------------------
            ### start training generator

            for e in range(start_epoch, args.n_epochs_G+1):
                if has_wandb:
                    wandb.log({"epoch": e})

                train_G(args, idx, net, generator, teacher, e, 
                            optimizer_S, optimizer_G, criterion, 
                            lim_0, lim_1, # mean_sample, var_sample,
                            loss_r_feature_layers)
        
                # #||||||||||||||||||||||||
                # # print(aa)
                # mean_layers_check = dict(zip(name_layer, aa.mean_layers))
                # mean_layers_check = collections.OrderedDict(mean_layers_check)
                # for i in mean_layers_check:
                #     print(i, mean_layers_check[i].shape)
                # #||||||||||||||||||||||||


                # test(args, net, data_test_loader, criterion)
            
                save_name = "start-"+str(start_class)+"_end-"+str(end_class)+".pth"
                torch.save({'epoch': e,
                            'G_state_dict': generator.state_dict(),
                            'G_optimizer_state_dict':optimizer_G.state_dict()}, 
                            save_path+"/"+save_name)

    # ------------------------------------------------
    ### train student
    if args.train_S:

        start_epoch = args.n_epochs_G

        # ------------------------------------------------
        ### load teachers
        G_list = []
        num_G = int(args.n_divid)

        if args.dataset == 'cifar10':
            num_classes = int(10/num_G)
        elif args.dataset == 'cifar100':
            num_classes = int(100/num_G)

        for i in range(0, num_G):
        # for i in range(0, 3):

            # start_class = i*num_classes
            # end_class = (i+1)*num_classes

            start_class = i
            end_class = i+1

            generator = Generator() # .cuda()
            generator = nn.DataParallel(generator)

            # -----------
            G_name = "start-"+str(start_class)+"_end-"+str(end_class)+".pth"
            print(save_path+'/'+G_name)
            ckeckpoints = torch.load(save_path+'/'+G_name)
            generator.load_state_dict(ckeckpoints['G_state_dict'])
            generator.eval()
            G_list.append(generator)

        print(">>>>> Finish Loading Generators")

        # ------------------------------------------------
        if args.dataset == 'cifar10':
            print("!!!! CIFAR-10")
            _, data_test_loader = get_split_cifar10(args, args.batch_size, 0, 10)

            net = resnet.ResNet18().cuda()
            criterion = torch.nn.CrossEntropyLoss().cuda()
            optimizer_S = torch.optim.SGD(net.parameters(), lr=args.lr_S, momentum=0.9, weight_decay=5e-4)

        if args.dataset == 'cifar100':
            print("!!!! CIFAR-100")
            _, data_test_loader = get_split_cifar100(args, args.batch_size, 0, 100)

            net = resnet.ResNet18(num_classes=100).cuda()
            criterion = torch.nn.CrossEntropyLoss().cuda()
            optimizer_S = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

        # ------------------------------------------------
        if args.resume:
            load_name = "{}_trainS_{}_ld{}_eN{}_eG{}_lrG{}_lrS{}.pth".format(args.dataset,
                                                                                args.fix_G, args.latent_dim,
                                                                                args.n_epochs, args.n_epochs_G,
                                                                                args.lr_G, args.lr_S)
            print("!!!! RESUME !!!!")
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

            # losses = [[] for _ in range(len(G_list))]
            # accuracy = [[] for _ in range(len(G_list))]

            if args.dataset != 'MNIST':
                adjust_learning_rate(args, optimizer_S, e)

            # print("111111111")
            train_S(args, net, G_list, teacher, e, optimizer_S)

            # ### Checking student accuracy
            # print(">>> Checking student accuracy")
            # test_S(args, net, len(G_list), num_classes, criterion)

            #### save student model
            print("-------> Model saved!!")
            save_name = "{}_trainS_{}_ld{}_eN{}_eG{}_lrG{}_lrS{}.pth".format(args.dataset, 
                                                                                args.fix_G, args.latent_dim,
                                                                                args.n_epochs, args.n_epochs_G,
                                                                                args.lr_G, args.lr_S)
            # save_name = "student.pth"
            torch.save({'epoch': e,
                        'S_state_dict': net.state_dict(),
                        'S_optimizer_state_dict': optimizer_S.state_dict()},
                        save_path+"/"+save_name)
    
            test(args, net, data_test_loader, criterion)

#############################################################
if __name__ == '__main__':
    main()
