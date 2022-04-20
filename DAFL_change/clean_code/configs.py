import argparse
import os

try:
    import wandb
    has_wandb = True
except ImportError: 
    has_wandb = False

imagenet_path = '/data/imagenet'
cache_path = '../cache'

def get_args():

    parser = argparse.ArgumentParser(description='train-teacher-network')

    # Basic model parameters.
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['MNIST','cifar10','cifar100', 'tiny'])
    parser.add_argument('-a', '--arch', metavar='ARCH', default='vgg16', choices=["resnet34", "vgg16"], help="teacher model")
    parser.add_argument('-a_s', '--arch_s', metavar='ARCH', default='vgg', choices=["resnet", "vgg"], help="student model")
    parser.add_argument('--data', type=str, default=cache_path + '/data/')

    parser.add_argument('--imagenet_path', type=str, default=imagenet_path)
    parser.add_argument('-r', "--ratio", type=float, default=0.5, help='use gen threshold ratio, 1 all gan')

    parser.add_argument('--save_path', default='/tmp')
    parser.add_argument('--stat_path', default='../stats/')

    parser.add_argument('--teacher_dir', type=str, default=cache_path + '/models/')
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

    parser.add_argument('--disable_wandb', action='store_true', help='disable wandb')
    
    parser.add_argument('--Gindex', type=int, help='index of generator')

    args = parser.parse_args()


    args.save_path = cache_path + '/ckpts_'+args.dataset+"_"+args.arch+'/multi_'+args.ext
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    return args



def set_logger(args):
    if has_wandb and not args.disable_wandb:
        if args.train_G:
            id = f"D-trainG-{args.dataset}{args.arch}-bz{args.batch_size}-{args.ext}"
        if args.train_S:
            id = f"D-trainS-{args.dataset}{args.arch}{args.arch_s}-r{args.ratio}lrS{args.lr_S}bz{args.batch_size}-{args.ext}"
        
        # if "asimov" in os.environ["$HOSTNAME"]:
        # wandb.init(project='few-shot-multi', entity='tidedancer', config=args, resume="allow", id=id)
        # else:
        wandb.init(project='few-shot-multi', entity='zhoushanglin100', config=args)
        wandb.config.update(args)
        log_func = wandb.log
    
    else:
        log_func = print

    return log_func


# if __name__ == '__main__':
#     args = get_args()
#     log_func = set_logger(args)
#     print(log_func)
