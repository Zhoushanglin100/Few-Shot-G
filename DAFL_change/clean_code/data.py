import torchvision
import torch
from torch.utils.data import Dataset, DataLoader


class BaseDataLoader():
    def __init__(self):
        self.trainset = None
        self.testset = None

    def _transform(self):
        raise NotImplementedError

    def build_trainloader(self, batch_size=128):
        return DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    def build_testloader(self, batch_size=100):
        return DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=4)


class CifarDataLoader(BaseDataLoader):
    def __init__(self, name='cifar10', cache_path='../cache/data/'):
        transform_train, transform_test = self._transform()
        if name == 'cifar10':
            self.trainset = torchvision.datasets.CIFAR10(cache_path, train=True, download=True, transform=transform_train)
            self.testset = torchvision.datasets.CIFAR10(cache_path, train=False, download=True, transform=transform_test)
        elif name == 'cifar100':
            self.trainset = torchvision.datasets.CIFAR100(cache_path, train=True, download=True, transform=transform_train)
            self.testset = torchvision.datasets.CIFAR100(cache_path, train=False, download=True, transform=transform_test)

    def _transform(self):
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
        return transform_train, transform_test


class ImagenetDataLoader(BaseDataLoader):
    def __init__(self, imagenet_path='/data/imagenet/train'):
        transform = self._transform()
        self.trainset = torchvision.datasets.ImageFolder(imagenet_path, transform=transform)

    def _transform(self):
        transform_train = torchvision.transforms.Compose([
                                    torchvision.transforms.Resize(size = (32,32)),
                                    torchvision.transforms.RandomHorizontalFlip(),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])
        return transform_train
    

# class TinyimagenetDataLoader(BaseDataLoader):
#     def build_testloader(self, args, batch_size=100):
#         DATA_DIR = "/data/tiny-imagenet-200"
#         _, data_test_loader = get_split_TinyImageNet(args, DATA_DIR, batch_size, 0, 200)
#         return data_test_loader


# class GeneratorDataset(Dataset):
#     def __init__(self, generator_list, latent_dim, cache_size=256):
#         # torch.multiprocessing.set_start_method('spawn')
#         self.generator_list = generator_list
#         self.num_gens = len(self.generator_list)
#         self.latent_dim = latent_dim
#         self.cache_size = cache_size
#         self.idx = 0
#         self._refresh()

#     def _refresh(self):
#         self.imgs = []
#         self.gids = []
#         for idx, generator in enumerate(self.generator_list):
#             generator.eval()
#             z = torch.randn(self.cache_size, self.latent_dim, requires_grad=False)
#             self.imgs.append(generator(z).detach().cpu())
#             self.gids.append(torch.tensor([idx]).repeat(self.cache_size))
#         self.imgs = torch.cat(self.imgs)[torch.randperm(self.cache_size * self.num_gens)]
#         self.gids = torch.cat(self.gids)[torch.randperm(self.cache_size * self.num_gens)]

#     def __len__(self):
#         return 999999

#     def __getitem__(self, idx):
#         print(self.idx)
#         img, gid = self.imgs[self.idx], self.gids[self.idx]
#         self.idx += 1
#         if self.idx == self.cache_size * self.num_gens:
#             self.idx = 0
#             self._refresh()
#         return img, gid.item() # convert gid tensor to an int, compatible with imagenet


# class MixDataset(Dataset):
#     def __init__(self, generator_list, latent_dim, cache_size, gen_ratio):
#         self.gen_dataset = GeneratorDataset(generator_list, latent_dim, cache_size)
#         self.imagenet_dataset = ImagenetDataLoader().trainset
#         self.gen_ratio = gen_ratio

#     def __len__(self):
#         return len(self.imagenet_dataset)

#     def __getitem__(self, idx):
#         if torch.rand(1).item() < self.gen_ratio:
#             return self.gen_dataset.__getitem__(idx)
#         else:
#             return self.imagenet_dataset.__getitem__(idx)


# class MixDataLoader(BaseDataLoader):
#     def __init__(self, generator_list, latent_dim, cache_size=256, gen_ratio=0.5):
#         self.trainset = MixDataset(generator_list, latent_dim, cache_size, gen_ratio)


# if __name__ == '__main__':
#     # # --------- test cifar or imagenet dataset --------
#     # # cifar10_loader = CifarDataLoader('cifar10').build_trainloader()
#     # imagenet_loader = ImagenetDataLoader().build_trainloader()
#     # for batch_idx, (inputs, _) in enumerate(imagenet_loader):
#     #     print(batch_idx)
#     #     print(inputs.shape)
#     #     if batch_idx > 30:
#     #         break

#     # -------- test generator mix dataset --------

#     import models
#     G_list = []
#     for i in range(0, 3):
#         start_class = i
#         end_class = i+1
#         generator = models.Generator(32, 3, 5000)
#         G_name = "start-"+str(start_class)+"_end-"+str(end_class)+".pth"
#         #print('/home/shz15022/DAFL_change/cache/ckpts_cifar10_vgg16/multi_allNORMsz10ld5000lr0.1ep100itr500/'+G_name)
#         ckeckpoints = torch.load('/home/shz15022/DAFL_change/cache/ckpts_cifar10_vgg16/multi_allNORMsz10ld5000lr0.1ep100itr500/'+G_name)
#         generator.load_state_dict(ckeckpoints['G_state_dict'])
#         generator =torch.nn.DataParallel(generator).cuda()
#         generator.eval()
#         G_list.append(generator)

#     mix_loader = MixDataLoader(G_list, 5000).build_trainloader()
#     print('dataset length', len(mix_loader))
#     for batch_idx, (inputs, label) in enumerate(mix_loader):
#         print(batch_idx)
#         print(inputs.shape)
#         print(label)
#         if batch_idx > 1000:
#             break