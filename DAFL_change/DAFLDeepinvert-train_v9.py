import os, random
from model.resnet import *
from model.vgg_block import vgg_stock, vgg_bw, cfgs, split_block
import torch
from torch.autograd import Variable
import argparse

from script import *

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
parser.add_argument('--dataset', type=str, default='cifar10', choices=['MNIST','cifar10','cifar100', 'tiny'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='vgg16', choices=["resnet34", "vgg16"], help="teacher model")
parser.add_argument('-a_s', '--arch_s', metavar='ARCH', default='vgg', choices=["resnet", "vgg"], help="student model")
parser.add_argument('--data', type=str, default='cache/data/')

parser.add_argument('--imagenet_path', type=str, default='/data/imagenet/')
parser.add_argument('--r', type=float, default=0.5, help='use gen threshold ratio, 1 all gan')

parser.add_argument('--output_dir', type=str, default='cache/models/')
parser.add_argument('--teacher_dir', type=str, default='cache/models/')
parser.add_argument('--ext', type=str, default='')
parser.add_argument('--resume', action='store_true', default=False,
                    help='whether to resume from ckpt')

parser.add_argument('--train_G', action='store_true', default=False,
                    help='whether to train multiple generators')
parser.add_argument('--train_S', action='store_true', default=False,
                    help='whether to train student')

parser.add_argument('--n_epochs_G', type=int, default=50, help='number of epochs of training generator')
parser.add_argument('--n_epochs', type=int, default=2000, help='number of epochs of training total')

parser.add_argument('--fix_G', action='store_true', default=False,
                    help='whether stop train generator after start training student')

parser.add_argument('--hook_type', type=str, default='output', choices=['input', 'output'],
                    help = "hook statistics from input data or output data")
parser.add_argument('--stat_type', type=str, default='extract', choices=['running', 'extract'],
                    help = "statistics from self extracted from a batch or saved stats from teacher")
parser.add_argument('--stat_bz', type=int, default=1, help='size of the batches')
parser.add_argument('--stat_layer', type=str, default='all', choices=['multi', 'single', 'convbn', "all"])
parser.add_argument('--data_type', type=str, default='sample', choices=['everyclass', 'sample'])

parser.add_argument('--batch_size', type=int, default=128, help='size of the batches')
parser.add_argument('--lr_G', type=float, default=0.001, help='learning rate of generator')
parser.add_argument('--lr_S', type=float, default=0.06, help='learning rate of student')
parser.add_argument('--decay', type=float, default=5, help='decay of learning rate')

parser.add_argument('--lambda_s', type=int, default=10, 
                    help='coefficient for moment matching loss. [cifar10: 10; cifar100: 1; imagenet: 3]')
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
        id = f"New-{args.dataset}{args.arch}-trainG-{args.ext}"
    if args.train_S:
        id = f"New-img32-{args.dataset}{args.arch}{args.arch_s}-trainS-r{args.r}lrS{args.lr_S}bz{args.batch_size}-{args.ext}"
    # if "asimov" in os.environ["$HOSTNAME"]:
    wandb.init(project='few-shot-multi', entity='tidedancer', config=args, resume="allow", id=id)
    # else:
    # wandb.init(project='few-shot-multi', entity='zhoushanglin100', config=args)#, resume="allow", id=id)
    wandb.config.update(args)

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
        self.var_layers = []

        # print(self.mean_dict.keys(), self.var_dict.keys())

    def hook_fn(self, module, input, output):

        # hook co compute deepinversion's feature distribution regularization
        if self.hook_type == "input":
            feature_map = input[0]
        elif self.hook_type == "output":
            feature_map = output
            
        nch = feature_map.shape[1]
        if len(feature_map.shape) == 4:
            mean = feature_map.mean([0,2,3])
            var = feature_map.permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        elif len(feature_map.shape) == 2:
            mean = feature_map.mean([0]).cpu().detach()
            var = feature_map.permute(1, 0).contiguous().view([nch, -1]).var(1, unbiased=False).cpu().detach()
        
        self.mean_layers.append(mean)
        self.var_layers.append(var)

        if self.stat_type == "running":
            # forcing mean and variance to match between two distributions
            # other ways might work better, e.g. KL divergence
            r_feature = torch.norm(module.running_var.data.type(var.type()) - var, 2) + torch.norm(module.running_mean.data.type(mean.type()) - mean, 2)
        elif self.stat_type == "extract":
            batch_mean = self.mean_dict[self.name].cuda()
            batch_var = self.var_dict[self.name].cuda()

            # criterion = nn.CosineEmbeddingLoss()
            # r_feature = criterion(batch_var.view(1,-1), var.view(1,-1), torch.ones(1).cuda()) + criterion(batch_mean.view(1,-1), mean.view(1,-1), torch.ones(1).cuda())

            r_feature = torch.norm(batch_var.type(var.type()) - var, 2) + torch.norm(batch_mean.type(mean.type()) - mean, 2)

            # print(r_feature.item())
            # print("v", criterion(batch_var.view(1,-1), var.view(1,-1), torch.ones(1).cuda()).item())
            # print("m", criterion(batch_mean.view(1,-1), mean.view(1,-1), torch.ones(1).cuda()).item())

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
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            # nn.BatchNorm2d(64),
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
        lr = args.lr_S * (epoch-args.n_epochs_G) / 10
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
    l_kl = F.kl_div(p, q, reduction='sum')  / y.shape[0]
    return l_kl

######################################################

def train_G(args, idx, net, generator, teacher, epoch,
            optimizer_S, optimizer_G, criterion, 
            lim_0, lim_1, # mean, var,
            loss_r_feature_layers): 

    # print("\n>>>>> Train Generators <<<<<\n")

    if args.dataset != 'MNIST':
        adjust_learning_rate_G(args, optimizer_G, epoch)
    
    net.train()
    loss = None

    num_itr = 500
    for i in range(num_itr):
    # for i in range(200):
    # for i in range(3):

        z = Variable(torch.randn(args.batch_size, args.latent_dim)).cuda()

        optimizer_G.zero_grad()

        gen_imgs = generator(z)

        ### one-hot loss
        if args.arch == "resnet34":
            outputs_T, features_T = teacher(gen_imgs, out_feature=True)
        else:
            outputs_T = teacher(gen_imgs)
        pred = outputs_T.data.max(1)[1]
        loss_one_hot = criterion(outputs_T,pred)
        softmax_o_T = torch.nn.functional.softmax(outputs_T, dim = 1).mean(dim = 0)
        loss_information_entropy = (softmax_o_T * torch.log10(softmax_o_T)).sum()
        # loss_activation = -features_T.abs().mean()
        # loss = loss_one_hot * args.oh + loss_information_entropy * args.ie + loss_activation * args.a 
        
        ### KD loss
        if args.arch_s == "resnet":
            outputs_S, features_S = net(gen_imgs, out_feature=True)
        else:
            outputs_S = net(gen_imgs)
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

        ### R_feature loss
        loss_distr = sum([mod.r_feature for mod in loss_r_feature_layers])

        ### only train generator before n_epochs_G epoch
        loss = loss_one_hot
        loss += (6e-3 * loss_var)
        loss += (1.5e-5 * torch.norm(gen_imgs, 2))  # l2 loss
        loss += int(args.lambda_s)*loss_distr                 # best for noise before BN

        if i % 100 == 0:
            print('Train G_%d, Epoch %d, Batch: %d, Loss: %f' % (idx, epoch, i, loss.data.item()))

            if has_wandb:
                loss_dict = {"loss_G/OneHot_Loss_"+str(idx): loss_one_hot.item(), 
                             "loss_G/KD_Loss_S_"+str(idx): loss_kd.item(),
                             "loss_G/Var_Loss_"+str(idx): loss_var.item(),
                             "loss_G/R_Loss_"+str(idx): loss_distr,
                             "loss_G/L2_Loss_"+str(idx): torch.norm(gen_imgs, 2).item(),
                             "G_perf/total_loss_"+str(idx): loss.data.item()
                            }
                wandb.log(loss_dict, step=i+epoch*500)

                # wandb.log({"loss_G/OneHot_Loss_"+str(idx): loss_one_hot.item()}, step=i)
                # wandb.log({"loss_G/KD_Loss_S_"+str(idx): loss_kd.item()}, step=i)
                # wandb.log({"loss_G/Var_Loss_"+str(idx): loss_var.item()}, step=i)
                # # wandb.log({"loss_G/R_Loss_"+str(idx): loss_distr.item()})
                # wandb.log({"loss_G/R_Loss_"+str(idx): loss_distr}, step=i)
                # wandb.log({"loss_G/L2_Loss_"+str(idx): torch.norm(gen_imgs, 2).item()}, step=i)
                # wandb.log({"G_perf/total_loss_"+str(idx): loss.data.item()}, step=i)

        loss.backward()
        optimizer_G.step()


# --------------------


def Check_train_S(args, net, data_train_loader, teacher, epoch, optimizer_S):
    print("\n>>>>> Train Student (check) <<<<<")

    if args.dataset != 'MNIST':
        adjust_learning_rate(args, optimizer_S, epoch)
    
    net.train()

    for i, (inputs, targets) in enumerate(data_train_loader):

        loss_total = None
        optimizer_S.zero_grad()

        if args.arch == "resnet34":
            outputs_T, features_T = teacher(inputs, out_feature=True)
        else:
            outputs_T = teacher(inputs)

        if args.arch_s == "resnet":
            outputs_S, features_S = net(inputs, out_feature=True)
        else:
            outputs_S = net(inputs)
        loss_kd = kdloss(outputs_S, outputs_T)

        ### only train student after n_epochs_G epochs
        if args.fix_G:
            loss = loss_kd
        else:
            print("To-Do")

        if i % 100 == 0:
            print('Student Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.data.item()))
            if has_wandb:
                wandb.log({"check_total_loss_S": loss.item()}, step=i+epoch*len(data_train_loader))

        loss.backward()
        optimizer_S.step()



def train_S(args, net, G_list, teacher, epoch, optimizer_S, imagenet_train_loader):
    print("\n>>>>> Train Student <<<<<")

    if args.dataset != 'MNIST':
        adjust_learning_rate(args, optimizer_S, epoch)
    
    net.train()

    # for i in range(500):
    # for i in range(200):
    # for i in range(5):
    num_itr = 500
    for batch_idx, (imagenet_input, _) in enumerate(imagenet_train_loader):
        if batch_idx > num_itr:
            break

        imagenet_input = imagenet_input.cuda()

        loss_total = None
        optimizer_S.zero_grad()

        if torch.rand(1).item() > args.r:
            gen_imgs = imagenet_input
        else:
            for gidx, generator in enumerate(G_list):
                generator = generator.cuda()

                ### KD loss
                z = Variable(torch.randn(args.batch_size, args.latent_dim)).cuda()
                gen_imgs_gid = generator(z)
                if gidx == 0:
                    gen_imgs = gen_imgs_gid
                else:
                    gen_imgs = torch.cat((gen_imgs, gen_imgs_gid), 0)

        if args.arch == "resnet34":
            outputs_T, features_T = teacher(gen_imgs, out_feature=True)
        else:
            outputs_T = teacher(gen_imgs)

        if args.arch_s == "resnet":
            outputs_S, features_S = net(gen_imgs, out_feature=True)
        else:
            outputs_S = net(gen_imgs)
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

        if batch_idx % 100 == 0:
            print('Student Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, batch_idx, loss.data.item()))
            if has_wandb:
                wandb.log({"total_loss_S": loss.item()}, step=batch_idx+epoch*num_itr)
        loss.backward()
        optimizer_S.step()
        
# ------------------------------------

def test(args, net, data_test_loader, criterion):

    net.eval()
    total_correct = 0
    avg_loss = 0.0
    total_len = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            output = net(images)
            avg_loss += criterion(output, labels).sum()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
            total_len += labels.size(0)

    avg_loss /= len(data_test_loader)
    acc = float(total_correct) / total_len
        
    print('\n|||| Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), acc))
    
    if has_wandb:
        wandb.log({"test_loss": avg_loss.data.item(), "test_acc": acc})


# ------------------------------------

# def test_S(args, net, len_G, num_classes, criterion):

#     net.eval()

#     losses = [0 for _ in range(len_G)]
#     correct = [0 for _ in range(len_G)]
#     total = [0 for _ in range(len_G)]

#     with torch.no_grad():

#         # batch_idx = 0
        
#         for i in range(len_G):

#             start_class = i*num_classes
#             end_class = (i+1)*num_classes
#             # print("test_S start_class: "+str(start_class)+" end_class: "+str(end_class))

#             if args.dataset == 'cifar10':
#                 _, test_loader = get_split_cifar10(args, args.batch_size, start_class, end_class*(i+1))
#             elif args.dataset == 'cifar100':
#                 _, test_loader = get_split_cifar100(args, args.batch_size, start_class, end_class*(i+1))
#             elif args.dataset == 'tiny':
#                 DATA_DIR = "/data/tiny-imagenet-200"
#                 _, test_loader = get_split_TinyImageNet(args, DATA_DIR, args.batch_size, start_class, end_class*(i+1))

#             for images, labels in test_loader:
#                 images, labels = Variable(images).cuda(), Variable(labels).cuda()
#                 outputs = net(images)
#                 loss = criterion(outputs, labels).sum()

#                 losses[i] += loss.item()
#                 pred = outputs.data.max(1)[1]
#                 correct[i] += pred.eq(labels.data.view_as(pred)).sum().item()
#                 total[i] += labels.size(0)

#             # print('Generator {}:'.format(i + 1))
#             # print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (losses[i]/(1*(batch_idx+1)), 100.*correct[i]/total[i], correct[i], total[i]))

#             loss_tmp = losses[i]/len(test_loader)
#             acc_tmp = 100. * correct[i] / total[i]

#             if has_wandb:
#                 wandb.log({"S_perf_test/S_test_loss_"+str(i+1): loss_tmp})
#                 wandb.log({"S_perf_test/S_test_acc_"+str(i+1): acc_tmp})

#     acc_total = 100. * sum(correct) / sum(total)
#     if has_wandb:
#         wandb.log({"S_test_acc_total": acc_total})

#     # print('>>>>> Total:')
#     # print('Acc: %.3f%% (%d/%d)' % (100* sum(correct) / sum(total), sum(correct), sum(total)))

#############################################################

def main():

    global acc, acc_best
    os.makedirs(args.output_dir, exist_ok=True)  

    acc = 0
    acc_best = 0
    start_epoch = 1

    # -----------------------------------------------
    ### load teacher
    print(f"Dataset: {args.dataset}, teacher: {args.arch}")
    if args.dataset == "cifar10":
        if args.arch == "resnet34":
            teacher = torch.load(args.teacher_dir + 'teacher_acc_95.3')
        elif args.arch == "vgg16":
            teacher = vgg_stock(cfgs['vgg16'], args.dataset, 10)
            checkpoint = torch.load('cache/models/vgg16_CIFAR10_ckpt.pth')
            teacher.load_state_dict(checkpoint['net'])
    elif args.dataset == "cifar100":
        if args.arch == "resnet34":    
            teacher = ResNet34(num_classes=100)
            ckpt_teacher = torch.load("cache/pretrained/cifar100_resnet34.pth")    # 74.41%
            teacher.load_state_dict(ckpt_teacher['state_dict'])
        elif args.arch == "vgg16":
            teacher = vgg_stock(cfgs['vgg16'], args.dataset, 100)
            checkpoint = torch.load('cache/models/vgg16_CIFAR100_ckpt.pth')
            teacher.load_state_dict(checkpoint['net'])
    elif args.dataset == "tiny":
        teacher = ResNet34(num_classes=200)
        file_name = "cache/models/tinyimagenet_resnet34.pth"
        teacher.load_state_dict(torch.load(file_name))
    else:
        teacher = models.resnet34(pretrained=True)

    teacher.cuda()
    teacher = nn.DataParallel(teacher)
    
    # -------------------------------------
    save_path = 'cache/ckpts_'+args.dataset+"_"+args.arch+'/multi_'+args.ext
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.stat_layer == "multi":
        stat_path = "stats/stats_"+args.dataset+"_"+args.arch+"/stats_multi_"+args.data_type+"_splz"+str(args.stat_bz)+"/"+args.hook_type
    elif args.stat_layer == "single":
        stat_path = "stats/stats_"+args.dataset+"_"+args.arch+"/stats_single_"+args.data_type+"_splz"+str(args.stat_bz)+"/"+args.hook_type
    elif args.stat_layer == "convbn":
        stat_path = "stats/stats_"+args.dataset+"_"+args.arch+"/stats_CBN_multi_"+args.data_type+"_splz"+str(args.stat_bz)+"/"+args.hook_type
    elif args.stat_layer == "all":
        stat_path = "stats/stats_"+args.dataset+"_"+args.arch+"/stats_all_multi_"+args.data_type+"_splz"+str(args.stat_bz)+"/"+args.hook_type
    print("Stats path:", stat_path)
    n_divid = int(len(os.listdir(stat_path))/2)
    print("!!!!!!!!", n_divid, "!!!!!!!!!!!!!")

    # ------------------------------------------------
    ### train generator
    if args.train_G:
        # setting up the range for jitter
        lim_0, lim_1 = 2, 2
        n = int(n_divid)

        ### iteratively train generators
        for idx in range(0, n):
            
            start_class = idx
            end_class = idx+1

            print("\n !!!!! start_class: "+str(start_class)+" end_class: "+str(end_class))

            # ---------------
            ### way to resume generator
            save_name = "start-"+str(start_class)+"_end-"+str(end_class)+".pth"

            if os.path.exists(save_path+"/"+save_name):
                ckeckpoints = torch.load(save_path+"/"+save_name)
                if ckeckpoints["epoch"] == args.n_epochs_G:
                    print("Generate exits!!", save_name)

                    continue
                    
            # ------------------------------------------------
            
            generator = Generator().cuda()
            generator = nn.DataParallel(generator)

            ### optimization
            criterion = torch.nn.CrossEntropyLoss().cuda()
            optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr_G)
            
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
    
            net.cuda()
            optimizer_S = torch.optim.SGD(net.parameters(), lr=args.lr_S, momentum=0.9, weight_decay=5e-4)

            ### Create hooks for feature statistics catching
            mean_layers_dictionary = torch.load(stat_path+"/mean_"+args.arch+"_start-"+str(start_class)+"_end-"+str(end_class)+".pth")
            var_layers_dictionary = torch.load(stat_path+"/var_"+args.arch+"_start-"+str(start_class)+"_end-"+str(end_class)+".pth")

            # ----------------------------------------------------
            # print("\n||||||||||||||")
            # for name_stat, module_stat in teacher.named_modules():
            #     if (isinstance(module_stat, nn.Conv2d) and (args.arch == "vgg16")) or ((args.arch == "resnet34") or ("bn" in name_stat) or ("downsample.1" in name_stat)):
            #         print(name_stat)
            # print("------------------")
            # for i in mean_layers_dictionary:
            #     print(i, mean_layers_dictionary[i].shape, var_layers_dictionary[i].shape)
            # print("------------------")
            # print("mean_layers_dictionary", mean_layers_dictionary.keys())
            # print("\nvar_layers_dictionary", var_layers_dictionary.keys())
            # print("||||||||||||||\n")
            # exit(0)
            # ----------------------------------------------------

            mean_layers_dictionary = {f'module.{k}': v for k, v in mean_layers_dictionary.items()}
            var_layers_dictionary = {f'module.{k}': v for k, v in var_layers_dictionary.items()}

            loss_r_feature_layers = []
            name_layer = []
            for name, module in teacher.named_modules():
                # if isinstance(module, nn.BatchNorm2d):
                # if (isinstance(module, nn.BatchNorm2d) and (args.arch == "vgg16")) or ((args.arch == "resnet34") and ("bn" in name) or ("downsample.1" in name)):
                # if ("bn" in name) or ("downsample.1" in name):
                if name in mean_layers_dictionary.keys():
                    aa = DeepInversionFeatureHook(args, name, module, mean_layers_dictionary, var_layers_dictionary)
                    loss_r_feature_layers.append(aa)
                    name_layer.append(name)

            # ------------------------------------------------
            ### start training generator

            for e in range(start_epoch, args.n_epochs_G+1):
            # for e in range(start_epoch, 2):

                if has_wandb:
                    wandb.log({"epoch": e})

                train_G(args, idx, net, generator, teacher, e, 
                            optimizer_S, optimizer_G, criterion, 
                            lim_0, lim_1,
                            loss_r_feature_layers)
            
                torch.save({'epoch': e,
                            'G_state_dict': generator.module.state_dict(),
                            'G_optimizer_state_dict':optimizer_G.state_dict()}, 
                            save_path+"/"+save_name)

            # ------------------------------------------------
            for aa in loss_r_feature_layers:
                aa.close()
            torch.cuda.empty_cache()

    # ------------------------------------------------
    ### train student
    if args.train_S:

        start_epoch = args.n_epochs_G

        # ------------------------------------------------
        ### load teachers
        num_G = int(n_divid)
        G_list = []

        for i in range(0, num_G):
        # for i in range(0, 3):

            start_class = i
            end_class = i+1

            # -----------
            generator = Generator()
            G_name = "start-"+str(start_class)+"_end-"+str(end_class)+".pth"
            print(save_path+'/'+G_name)
            ckeckpoints = torch.load(save_path+'/'+G_name)
            generator.load_state_dict(ckeckpoints['G_state_dict'])
            generator = nn.DataParallel(generator)
            generator.eval()
            G_list.append(generator)

            # start_epoch = max(start_epoch, ckeckpoints['e'])

        print(">>>>> Finish Loading Generators")

        # ------------------------------------------------
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

        if args.dataset == 'cifar10':
            print("!!!! CIFAR-10")
            # _, data_test_loader = get_split_cifar10(args, args.batch_size, 0, 10)
            trainset = torchvision.datasets.CIFAR10(args.data, train=True, download=True, transform=transform_train)
            data_train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
            testset = torchvision.datasets.CIFAR10(args.data, train=False, download=True, transform=transform_test)
            data_test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
            if args.arch_s == "resnet":
                net = ResNet18()
            elif args.arch_s == "vgg":
                net = vgg_bw(cfgs['vgg16-graft'], True, args.dataset, 10)
        if args.dataset == 'cifar100':
            print("!!!! CIFAR-100")
            # _, data_test_loader = get_split_cifar100(args, args.batch_size, 0, 100)
            trainset = torchvision.datasets.CIFAR100(args.data, train=True, download=True, transform=transform_train)
            data_train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
            testset = torchvision.datasets.CIFAR100(args.data, train=False, download=True, transform=transform_test)
            data_test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
            if args.arch_s == "resnet":
                net = ResNet18(num_classes=100)
            elif args.arch_s == "vgg":
                net = vgg_bw(cfgs['vgg16-graft'], True, args.dataset, 100)
        if args.dataset == 'tiny':
            print("!!!! Tiny ImageNet")
            DATA_DIR = "/data/tiny-imagenet-200"
            _, data_test_loader = get_split_TinyImageNet(args, DATA_DIR, args.batch_size, start_class, end_class)
            net = ResNet18(num_classes=200)
        net.cuda()
        criterion = torch.nn.CrossEntropyLoss().cuda()
        optimizer_S = torch.optim.SGD(net.parameters(), lr=args.lr_S, momentum=0.9, weight_decay=5e-4)
        # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer_S, start_factor=args.lr_S, total_iters=int(args.n_epochs-args.n_epochs_G))

        # ------------------------------------------------
        save_name = "imgNet_{}{}_{}_trainS_ld{}_eN{}_eG{}_lrG{}_lrS{}wp10_dcy{}_lambda{}.pth".format(args.dataset, args.arch_s,
                                                                            args.r, 
                                                                            args.latent_dim,
                                                                            args.n_epochs, args.n_epochs_G,
                                                                            args.lr_G, args.lr_S,
                                                                            args.decay, args.lambda_s)
        if args.resume:
            load_name = save_name
            print("!!!! RESUME !!!!")
            if os.path.exists(save_path+'/'+load_name):
                checkpoint = torch.load(save_path+'/'+load_name)
                net.load_state_dict(checkpoint['S_state_dict'])
                optimizer_S.load_state_dict(checkpoint['S_optimizer_state_dict'])
                resume_epoch = checkpoint['epoch']
                start_epoch = resume_epoch+1

        # ------------------------------------------------

        net = nn.DataParallel(net)

        for e in range(start_epoch, args.n_epochs):

            if has_wandb:
                wandb.log({"epoch": e})

            # ------------------------------------------------
            ### load imagenet for random sample
            imgnet_traindir = os.path.join(args.imagenet_path, 'train')
            imgnet_normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            imagenet_train_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(imgnet_traindir, 
                                                            torchvision.transforms.Compose([
                                                                # torchvision.transforms.RandomResizedCrop(32),
                                                                torchvision.transforms.Resize(size = (32,32)),
                                                                torchvision.transforms.RandomHorizontalFlip(),
                                                                torchvision.transforms.ToTensor(),
                                                                imgnet_normalize,
                                                            ])),
                                                        batch_size=args.batch_size, shuffle=True,
                                                        num_workers=4, pin_memory=True)
            # ------------------------------------------------

            # Check_train_S(args, net, data_train_loader, teacher, e, optimizer_S)
            train_S(args, net, G_list, teacher, e, optimizer_S, imagenet_train_loader)
            # scheduler.step()

            #### save student model
            # print("-------> Model saved!!")
            torch.save({'epoch': e,
                        'S_state_dict': net.module.state_dict(),
                        'S_optimizer_state_dict': optimizer_S.state_dict()},
                        save_path+"/"+save_name)
            
            # ### Checking student accuracy
            # print(">>> Checking student accuracy")
            test(args, net, data_test_loader, criterion)

#############################################################
if __name__ == '__main__':
    main()
