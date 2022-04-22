import torch
import torch.nn as nn
import os

### add deepinversion
class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, hook_type, stat_type, name, module, mean_dict, var_dict):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.name = name
        self.mean_dict = mean_dict
        self.var_dict = var_dict
        self.hook_type = hook_type
        self.stat_type = stat_type
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
            batch_mean = self.mean_dict[self.name]
            batch_var = self.var_dict[self.name]
            r_feature = torch.norm(batch_var.type(var.type()) - var, 2) + torch.norm(batch_mean.type(mean.type()) - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()


class Generator(nn.Module):
    def __init__(self, img_size=32, n_channels=3, latent_dim=5000):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.n_channels = n_channels
        self.latent_dim = latent_dim
        self.init_size = self.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(self.latent_dim, 128*self.init_size**2))

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
            nn.Conv2d(64, self.n_channels, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(self.n_channels, affine=False)
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

    @property
    def device(self):
        return next(self.parameters()).device


class GeneratorList(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_G = args.num_G
        self.load_path = args.save_path
        self.batch_size = args.batch_size
        self.latent_dim = args.latent_dim
        self.img_size = args.img_size
        self.channels = args.channels
        self.G_list = torch.nn.ModuleList([])
        self._get_generators()

    def _get_generators(self):
        for i in range(0, self.num_G):
            start_class = i
            end_class = i+1
            generator = Generator(img_size=self.img_size, n_channels=self.channels, latent_dim=self.latent_dim)
            G_name = "start-"+str(start_class)+"_end-"+str(end_class)+".pth"
            print(self.load_path+'/'+G_name)
            assert os.path.exists(self.load_path+'/'+G_name)
            ckeckpoints = torch.load(self.load_path+'/'+G_name, map_location=torch.device('cpu'))
            generator.load_state_dict(ckeckpoints['G_state_dict'])
            self.G_list.append(generator)
        print(">>>>> Finish Loading Generators")

    def forward(self):
        imgs = []
        gids = []
        num_gens = len(self.G_list)
        for idx, generator in enumerate(self.G_list):
            gen_size = round(self.batch_size / num_gens)
            if idx == num_gens-1:
                gen_size = self.batch_size - round(self.batch_size/num_gens)*(num_gens-1)
            z = torch.randn(gen_size, self.latent_dim, requires_grad=False, device=generator.device)
            imgs.append(generator(z))
            gids.append(torch.tensor([idx]).repeat(gen_size))
        idx = torch.randperm(self.batch_size)
        imgs = torch.cat(imgs)[idx]
        gids = torch.cat(gids)[idx]
        return imgs, gids
