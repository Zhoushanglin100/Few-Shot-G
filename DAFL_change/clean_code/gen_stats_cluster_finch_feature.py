import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
# from kmeans_pytorch import kmeans, kmeans_predict

from model.resnet import ResNet34
from model.vgg_block import vgg_stock, vgg_bw, cfgs, split_block
import model.resnet

import collections
from script import *

from finch import FINCH

import torchvision.models as models
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

#################################################################

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--dataset', type=str, default='cifar10', 
                    choices=['MNIST','cifar10','cifar100', 'tiny'])
parser.add_argument('--data-type', type=str, default='sample', 
                    choices=['everyclass', 'sample'])
parser.add_argument('--data', type=str, default='../cache/data/')
parser.add_argument('-a', '--arch', metavar='ARCH', default='vgg16',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--hook-type', type=str, default='output', choices=['input', 'output'],
                    help = "hook statistics from input data or output data")
parser.add_argument('--thrd', '--cluster-threshold', default=20, type=int, metavar='N',
                    help='maximum number of generators can train')
parser.add_argument('--stat-layer', type=str, default='all', 
                    choices=['multi', 'single', 'convbn', "all"])

parser.add_argument('--teacher-dir', type=str, default='../cache/models/')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=10, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

#################################################################
# print(parser)
torch.manual_seed(0)
best_acc1 = 0
mean_layers, var_layers = [], []

#################################################################

def main():
    args = parser.parse_args()
    print("------------------------")
    print(args)
    print("------------------------")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    ### create model
    if args.pretrained:
        # model = models.__dict__[args.arch](pretrained=True)
        if args.dataset == "cifar10":
            print("=> CIFAR10: using pre-trained model '{}'".format(args.arch))
            if args.arch == "resnet34":
                model = ResNet34(num_classes=10)
                model.load_state_dict(torch.load(args.teacher_dir + "cifar10_resnet34_95.3.pth"))
            elif args.arch == "vgg16":
                model = vgg_stock(cfgs['vgg16'], args.dataset, 10)
                checkpoint = torch.load(args.teacher_dir + 'vgg16_CIFAR10_ckpt.pth')
                model.load_state_dict(checkpoint['net'])
                # model.load_state_dict(torch.load("cache/models/vgg16-blockwise-cifar10.pth"))
        elif args.dataset == "cifar100":
            print("=> CIFAR100: using pre-trained model '{}'".format(args.arch))
            if args.arch == "resnet34":
                model = ResNet34(num_classes=100)
                ckpt_teacher = torch.load(args.teacher_dir + "cifar100_resnet34.pth")
                model.load_state_dict(ckpt_teacher['state_dict'])
            elif args.arch == "vgg16":
                model = vgg_stock(cfgs['vgg16'], args.dataset, 100)
                checkpoint = torch.load(args.teacher_dir + 'vgg16_CIFAR100_ckpt.pth')
                model.load_state_dict(checkpoint['net'])
                # model.load_state_dict(torch.load("cache/models/vgg16-blockwise-cifar100.pth"))
        elif args.dataset == "tiny":
            print("=> Tiny: using pre-trained model resnet34")
            # print(file_name)
            model = ResNet34(num_classes=200)
            file_name = "cache/models/tinyimagenet_resnet34.pth"
            model.load_state_dict(torch.load(file_name))
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
    # model = model.cuda()

    if not torch.cuda.is_available():
        print("111")
        print('using CPU, this will be slow')
    elif args.distributed:
        print("222")
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        print("333")
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        print("444")
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            print("444-1")
            # model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            print("444-2")
            # model = torch.nn.DataParallel(model).cuda()
            model = model.cuda()

    ### define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    ### optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # print(model)
    # print("\n||||||||||||||")
    # for name, W in model.named_parameters():
    #     if 'weight' in name:
    #     # if isinstance(name, nn.BatchNorm2d) or (isinstance(name, nn.Conv2d)):
    #         print(name, W.shape)
    # print("||||||||||||||\n")
    # exit(0)
    # for name_stat, module_stat in model.named_modules():
    #     if isinstance(module_stat,nn.Conv2d) or isinstance(module_stat,nn.BatchNorm2d) or isinstance(module_stat,nn.BatchNorm1d): 
    #         print(name_stat)

    cudnn.benchmark = True
    
    # #######################################################
    # # get stat of one batch --> store stat of BN
    # # check
    # if args.arch == "vgg16":
    #     temp = torch.load("stats/stats_cifar10_vgg16/stats_all_multi_sample_splz10/output/var_vgg16_start-0_end-1.pth")
    # elif args.arch == "resnet34":
    #     temp = torch.load("stats/stats_cifar10_resnet34/stats_multi_sample_splz10/output/mean_resnet34_start-0_end-1.pth")
    # for i in temp:
    #     print(i, temp[i].shape)
    #     # print(temp[i])
    # exit(0)
    # #######################################################

    ### Data loading
    if args.data_type == "sample":
        print("!!!!!!!!! SAMPLE")
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            ])

        if args.dataset == 'cifar10':
            trainset = torchvision.datasets.CIFAR10(args.data, train=True, download=True, transform=transform_train)        
        if args.dataset == 'cifar100':
            trainset = torchvision.datasets.CIFAR100(args.data, train=True, download=True, transform=transform_train)
        if args.dataset == 'Tiny':
            DATA_DIR = "/data/tiny-imagenet-200"
            TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
            preprocess_transform_pretrain = torchvision.transforms.Compose([
                        torchvision.transforms.CenterCrop(32),
                        torchvision.transforms.ToTensor(),  # Converting cropped images to tensors
                        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
            trainset = torchvision.datasets.ImageFolder(TRAIN_DIR, transform=preprocess_transform_pretrain)
        
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
        feature_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(next(iter(train_loader))[0], next(iter(train_loader))[1]), batch_size=args.batch_size)

    elif args.data_type == "everyclass":
        print("!!!!!!!!! EVERY CLASS")
        if args.dataset == 'cifar10':
            n = 10
            num_classes = int(10/n)
            train_images = None
            train_labels = None
            for idx in range(0, n):
                start_class = idx*num_classes
                end_class = (idx+1)*num_classes
                print("-----> start_class: "+str(start_class)+" end_class: "+str(end_class))
                train_loader, val_loader = get_split_cifar10(args, args.batch_size, start_class, end_class)
                train_inputs, train_classes = next(iter(train_loader))   
                if idx == 0:
                    train_images = train_inputs
                    train_labels = train_classes
                else:
                    train_images = torch.vstack((train_images, train_inputs))
                    train_labels = torch.cat((train_labels, train_classes))

        if args.dataset == 'cifar100':
            n = 100
            num_classes = int(100/n)
            train_images = None
            train_labels = None
            for idx in range(0, n):
                start_class = idx*num_classes
                end_class = (idx+1)*num_classes
                print("-----> start_class: "+str(start_class)+" end_class: "+str(end_class))
                train_loader, val_loader = get_split_cifar100(args, args.batch_size, start_class, end_class)
                train_inputs, train_classes = next(iter(train_loader))   
                if idx == 0:
                    train_images = train_inputs
                    train_labels = train_classes
                else:
                    train_images = torch.vstack((train_images, train_inputs))
                    train_labels = torch.cat((train_labels, train_classes))

        if args.dataset == 'tiny':
            DATA_DIR = "/data/tiny-imagenet-200"
            n = 200
            num_classes = int(200/n)
            train_images = None
            train_labels = None
            for idx in range(0, n):
                start_class = idx*num_classes
                end_class = (idx+1)*num_classes
                print("-----> start_class: "+str(start_class)+" end_class: "+str(end_class))
                train_loader, val_loader = get_split_TinyImageNet(args, DATA_DIR, args.batch_size, 
                                                                        start_class, end_class)
                train_inputs, train_classes = next(iter(train_loader))   
                if idx == 0:
                    train_images = train_inputs
                    train_labels = train_classes
                else:
                    train_images = torch.vstack((train_images, train_inputs))
                    train_labels = torch.cat((train_labels, train_classes))

        p = np.random.permutation(len(train_labels))
        train_images_shuffle, train_labels_shuffle = train_images[p], train_labels[p]
        feature_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_images_shuffle, train_labels_shuffle), batch_size=args.batch_size)

    # -----------------------------------

    linear_input, linear_output = validate(feature_loader, model, args)
    print(linear_input.shape)
    print("BZ=", args.batch_size)

    c, num_clust, req_c = FINCH(linear_input.detach().cpu().numpy(), distance='cosine', ensure_early_exit=True)   #  ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']]

    # # ------------------------------------
    # from sklearn.metrics import normalized_mutual_info_score as nmi
    # for i in range(len(num_clust)):
    #     acc = nmi(target_lst.detach().cpu().numpy(), c[:,i], average_method="max")
    #     print(acc)
    # # ------------------------------------

    cluster_ids, num_clusters = [g for g in enumerate(num_clust) if g[1] < args.thrd][-1]
    cluster_ids_train = torch.tensor(c[:, cluster_ids])

    print("\nFINCH, choose #cluster=", num_clusters, "\n")

    # ---------------------------------
    ### cluster
    # train_images_rsp = train_images.reshape(n*args.batch_size,-1)
    # val_images_rsp = val_images.reshape(val_images.shape[0],-1)

    ### kmeans
    # num_clusters = args.num_clusters
    # cluster_ids_train, cluster_centers = kmeans(X=train_images_rsp, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0'))
    # cluster_ids_val = kmeans_predict(val_images_rsp, cluster_centers, 'euclidean', device=torch.device('cuda:0'))
    # # print(cluster_ids_train, cluster_ids_val)
    
    # ### FINCH method
    # c, num_clust, req_c = FINCH(train_images_rsp.numpy(), distance='cosine')   #  ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']]
    # print(train_images_rsp.shape)
    # print("BZ=", args.batch_size)
    # num_clusters = num_clust[2]
    # cluster_ids_train = torch.tensor(c[:, 2])
    # print("\nFINCH, choose #cluster=", num_clusters, "\n")
    
    # exit(0)

    # ---------------------------------

    for idx_cluster in range(num_clusters):

        train_images = feature_loader.dataset.tensors[0]

        train_images_idx = train_images[torch.where(cluster_ids_train == idx_cluster)]
        train_label_idx = cluster_ids_train[torch.where(cluster_ids_train == idx_cluster)]-idx_cluster

        dataset_cluster = torch.utils.data.TensorDataset(train_images_idx, train_label_idx)
        train_loader_cluster = torch.utils.data.DataLoader(dataset_cluster, batch_size=len(train_label_idx))

        print(idx_cluster, train_images_idx.shape)
        
        # ---------------------------------

        for epoch in range(args.start_epoch, args.epochs):

            adjust_learning_rate(optimizer, epoch, args)

            ### train for one epoch
            train(args, train_loader_cluster, model, criterion, optimizer, epoch, idx_cluster, idx_cluster+1)

# -------------------------------------------------------------------------------------------------

def train(args, train_loader, model, criterion, optimizer, epoch, start_class, end_class):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    # ----------------------------------------
    name_layers = []
    def hook_fn_forward(module, input, output):
        
        if args.hook_type == "output":
            feature_map = output
        elif args.hook_type == "input":
            feature_map = input[0]
        # print("---->", feature_map.shape)

        nch = feature_map.shape[1]
        if len(feature_map.shape) == 4:
            mean = feature_map.mean([0,2,3]).cpu().detach()
            var = feature_map.permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False).cpu().detach()
        elif len(feature_map.shape) == 2:
            mean = feature_map.mean([0]).cpu().detach()
            var = feature_map.permute(1, 0).contiguous().view([nch, -1]).var(1, unbiased=False).cpu().detach()
 
        mean_layers.append(mean)
        var_layers.append(var)

    for name_stat, module_stat in model.named_modules():

        # print(module_stat)
        # if ((isinstance(module_stat, nn.BatchNorm2d) or isinstance(module_stat, nn.BatchNorm1d)) and (args.arch == "vgg16")) or ((args.arch == "resnet34") or ("bn" in name_stat) or ("downsample.1" in name_stat)):
        # if ("bn" in name_stat) or ("downsample.1" in name_stat):
        # if isinstance(module_stat,nn.Conv2d) or isinstance(module_stat,nn.BatchNorm2d): 
        if (args.stat_layer == "all") and (isinstance(module_stat,nn.Conv2d) or isinstance(module_stat,nn.BatchNorm2d) or isinstance(module_stat,nn.BatchNorm1d) or ("classifier.0" in name_stat)) \
            or (args.stat_layer == "convbn") and (isinstance(module_stat,nn.Conv2d) or isinstance(module_stat,nn.BatchNorm2d)) \
            or (args.stat_layer == "multi" or args.stat_layer == "single") and isinstance(module_stat,nn.Conv2d): 
            # print("--->", name_stat)
            module_stat.register_forward_hook(hook_fn_forward)
            name_layers.append(name_stat)
    # ----------------------------------------

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True).to(dtype=torch.long)

        output = model(images)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        ### construct dict to store mean and var
        mean_layers_dictionary, var_layers_dictionary = dict(zip(name_layers, mean_layers)), dict(zip(name_layers, var_layers))
        mean_layers_dictionary = collections.OrderedDict(mean_layers_dictionary)
        var_layers_dictionary = collections.OrderedDict(var_layers_dictionary)

        # for iii in mean_layers_dictionary:
        #     print(iii, mean_layers_dictionary[iii].shape)
        # print(mean_layers_dictionary.keys())

        ### save generated statistics
        save_path = "../stats/stats_"+args.dataset+"_"+args.arch+"/stats_"+args.stat_layer+"_"+args.data_type+"_splz"+str(args.batch_size)+"/"+args.hook_type
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(mean_layers_dictionary, save_path+"/mean_"+args.arch+"_"+"start-"+str(start_class)+"_end-"+str(end_class)+".pth")
        torch.save(var_layers_dictionary, save_path+"/var_"+args.arch+"_"+"start-"+str(start_class)+"_end-"+str(end_class)+".pth")

        if i % args.print_freq == 0:
            progress.display(i)

# ---------------------------------------------------------------------

def validate(val_loader, model, args):

    model.eval()

    with torch.no_grad():
        for i, (images, _) in enumerate(val_loader):

            images = images.cuda(args.gpu, non_blocking=True)

            # ------------------------
            res5c_input, res5c_output = None, None
            def res5c_hook(module, input_, output):
                nonlocal res5c_output
                nonlocal res5c_input
                res5c_output = output
                res5c_input = input_
            if args.arch == "resnet34":
                model.linear.register_forward_hook(res5c_hook)
            elif args.arch == "vgg16":
                model.classifier.register_forward_hook(res5c_hook)
            # ------------------------
            ### compute output (must have for hook) 
            output = model(images)

            if i == 0:
                res5c_input_lst = res5c_input[0]
                res5c_output_lst = res5c_output
            else:
                res5c_input_lst = torch.vstack((res5c_input_lst, res5c_input[0]))
                res5c_output_lst = torch.vstack((res5c_output_lst, res5c_output))

    return res5c_input_lst, res5c_output_lst

# ---------------------------------------------------------------------

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


#####################################################################

if __name__ == '__main__':
    main()
