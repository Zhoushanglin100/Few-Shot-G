import torch
import torch.nn as nn

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


### add gen
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
    

# if __name__ == '__main__':
#     import configs
#     args = configs.get_args()
#     gen = Generator()
#     hook = DeepInversionFeatureHook(args.hook_type, args.stat_type)
