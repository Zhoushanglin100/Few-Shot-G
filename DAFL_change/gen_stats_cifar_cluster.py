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
from kmeans_pytorch import kmeans, kmeans_predict

import resnet as resnet

import collections
from script import *

import torchvision.models as models
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

#################################################################

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--dataset', type=str, default='cifar10', 
                    choices=['MNIST','cifar10','cifar100'])
parser.add_argument('--data', type=str, default='cache/data/')
parser.add_argument('--n_divid', type=int, default=10, 
                    help='number of division of dataset')
parser.add_argument('--num_clusters', type=int, default=5, 
                    help='number of clusters')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--hook_type', type=str, default='output', choices=['input', 'output'],
                    help = "hook statistics from input data or output data")
parser.add_argument('--ext', type=str, default='')

parser.add_argument('--teacher_dir', type=str, default='cache/models/')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
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
            print("=> using pre-trained model '{}'".format(args.arch))
            model = torch.load(args.teacher_dir + 'teacher_acc_95.3')
        elif args.dataset == "cifar100":
            print("=> using pre-trained model '{}'".format(args.arch))
            model = resnet.ResNet34(num_classes=100).cuda()
            ckpt_teacher = torch.load("cache/pretrained/cifar100_resnet34.pth")
            model.load_state_dict(ckpt_teacher['state_dict'])
        elif args.dataset == "Tiny":
            print("=> using pre-trained model resnet34")
            model = models.resnet34(pretrained=True)
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
            model.features = torch.nn.DataParallel(model.features)
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
    #     if ('bn' in name) and ("weight" in name):
    #         print(name, W.shape)
    # print("||||||||||||||\n")

    # print(model.module.bn1.running_mean.data)
    # print(model.module.layer1[0].bn1.running_mean.data)
    # print(model.module.bn1.running_mean.data.shape, 
    #         model.module.layer1[0].bn1.running_mean.data.shape)
    # exit(0)

    cudnn.benchmark = True
    
    ### Data loading
    if args.dataset == 'cifar10':
        n = int(args.n_divid)
        num_classes = int(10/n)

        train_images = None
        train_labels = None

        # val_images = None
        # val_labels = None

        for idx in range(0, n):
            start_class = idx*num_classes
            end_class = (idx+1)*num_classes

            print("-----> start_class: "+str(start_class)+" end_class: "+str(end_class))

            train_loader, val_loader = get_split_cifar10(args, args.batch_size, 
                                                                    start_class, end_class)
            train_inputs, train_classes = next(iter(train_loader))   
            # val_inputs, val_classes = next(iter(train_loader))   

            if idx == 0:
                train_images = train_inputs
                train_labels = train_classes
                # val_images = val_inputs
                # val_labels = val_classes
            else:
                train_images = torch.vstack((train_images, train_inputs))
                train_labels = torch.cat((train_labels, train_classes))
                # val_images = torch.vstack((val_images, val_inputs))
                # val_labels = torch.cat((val_labels, val_classes))

    if args.dataset == 'cifar100':
        n = int(args.n_divid)
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

    if args.dataset == 'Tiny':

        DATA_DIR = "/data/tiny-imagenet-200"
        
        n = int(args.n_divid)
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

    # ---------------------------------
    ### cluster
    num_clusters = args.num_clusters

    train_images_rsp = train_images.reshape(n*args.batch_size,-1)
    # val_images_rsp = val_images.reshape(val_images.shape[0],-1)

    # kmeans
    cluster_ids_train, cluster_centers = kmeans(X=train_images_rsp, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0'))
    # cluster_ids_val = kmeans_predict(val_images_rsp, cluster_centers, 'euclidean', device=torch.device('cuda:0'))

    # print(cluster_ids_train, cluster_ids_val)
    
    for idx_cluster in range(num_clusters):

        train_images_idx = train_images[torch.where(cluster_ids_train == idx_cluster)]
        train_label_idx = cluster_ids_train[torch.where(cluster_ids_train == idx_cluster)]

        dataset_cluster = torch.utils.data.TensorDataset(train_images_idx, train_label_idx)
        train_loader_cluster = torch.utils.data.DataLoader(dataset_cluster)

        print(num_clusters, train_images_idx.shape, train_label_idx.shape)
        
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

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if args.gpu is not None:
            print('555-1')
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            print('555-2')
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

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
        
        #######################################################
        ### get stat of one batch --> store stat of BN

        # ### check
        # temp = torch.load("stats_multi/output/mean_resnet34_start-0_end-1.pth")
        # for i in temp:
        #     print(i, temp[i].shape)
        # exit(0)

        name_layers = []
        
        def hook_fn_forward(module, input, output):
            # first input dim ([64,64,112,112])
        
            if args.hook_type == "output":
                nch = output.shape[1]
                mean = output.mean([0,2,3]).cpu().detach()
                var = output.permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False).cpu().detach()
            elif args.hook_type == "input":   
                nch = input[0].shape[1]
                mean = input[0].mean([0, 2, 3]).cpu().detach()
                var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False).cpu().detach()    
            
            mean_layers.append(mean)
            var_layers.append(var)
        
        for name_stat, module_stat in model.named_modules():
            # print(name_stat)
            # print(module_stat)

            # if isinstance(module_stat, nn.BatchNorm2d):
            if ("bn" in name_stat) or ("downsample.1" in name_stat):
            #if len(module_stat.size()) == 4:
            #if isinstance(module_stat,nn.Conv2d): 
                module_stat.register_forward_hook(hook_fn_forward)
                name_layers.append(name_stat)
        # exit(0)
        #print(name_layers)
        #print(len(mean_layers))
        #exit(0)

        ### construct dict to store mean and var
        mean_layers_dictionary, var_layers_dictionary = dict(zip(name_layers, mean_layers)), dict(zip(name_layers, var_layers))
        mean_layers_dictionary = collections.OrderedDict(mean_layers_dictionary)
        var_layers_dictionary = collections.OrderedDict(var_layers_dictionary)

        # for iii in mean_layers_dictionary:
        #     print(iii, mean_layers_dictionary[iii].shape)
        # print(mean_layers_dictionary.keys())

        ### save generated statistics
        save_path = "stats_"+args.dataset+"/stats_multi_"+args.ext+"/"+args.hook_type
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(mean_layers_dictionary, save_path+"/mean_"+args.arch+"_"+"start-"+str(start_class)+"_end-"+str(end_class)+".pth")
        torch.save(var_layers_dictionary, save_path+"/var_"+args.arch+"_"+"start-"+str(start_class)+"_end-"+str(end_class)+".pth")

        if i % args.print_freq == 0:
            progress.display(i)
        if i == 1:
            return
        

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        ### TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


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
