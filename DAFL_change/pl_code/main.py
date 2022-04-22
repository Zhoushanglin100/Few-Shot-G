import pytorch_lightning as pl
import os
import torch
import torchvision
import utils
import configs
import data
from model.generator import Generator, GeneratorList

# class GT(pl.LightningModule):
#     def __init__(self, generator, teacher, optimizer, statistics, lr_scheduler, batch_size, latent_dim):
#         super().__init__()
#         self.generator = generator
#         self.teacher = teacher
#         self.statistics = statistics
#         self.optimizer = optimizer
#         self.lr_scheduler = lr_scheduler
#         self.batch_size = batch_size
#         self.latent_dim = latent_dim

#     def forward(self, batch):
#         imgs, labels = batch
#         output = self.student(imgs)
#         preds = output.max(dim=1)[1]
#         return preds

#     def training_step(self, batch, batch_idx):
#         z = torch.randn(self.batch_size, self.latent_dim, requires_grad=False, device=self.generator.device)
#         gen_imgs = self.generator(z)

#         ### one-hot loss
#         outputs_T = teacher(gen_imgs)
#         pred = outputs_T.data.max(1)[1]
#         loss_one_hot = criterion(outputs_T,pred)
#         # softmax_o_T = torch.nn.functional.softmax(outputs_T, dim = 1).mean(dim = 0)
#         # loss_information_entropy = (softmax_o_T * torch.log10(softmax_o_T)).sum()
#         # loss_activation = -features_T.abs().mean()
#         # loss = loss_one_hot * args.oh + loss_information_entropy * args.ie + loss_activation * args.a 
        
#         ### KD loss
#         outputs_S = net(gen_imgs)
#         loss_kd = utils.kdloss(outputs_S, outputs_T)

#         ### from deepinversion: variation loss
#         ## apply random jitter offsets
#         off1 = random.randint(-lim_0, lim_0)
#         off2 = random.randint(-lim_1, lim_1)
#         gen_imgs = torch.roll(gen_imgs, shifts=(off1, off2), dims=(2,3))

#         ## apply total variation regularization
#         diff1 = gen_imgs[:,:,:,:-1] - gen_imgs[:,:,:,1:]
#         diff2 = gen_imgs[:,:,:-1,:] - gen_imgs[:,:,1:,:]
#         diff3 = gen_imgs[:,:,1:,:-1] - gen_imgs[:,:,:-1,1:]
#         diff4 = gen_imgs[:,:,:-1,:-1] - gen_imgs[:,:,1:,1:]
#         loss_var = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)

#         ### R_feature loss
#         loss_distr = sum([mod.r_feature for mod in loss_r_feature_layers])

#         ### only train generator before n_epochs_G epoch
#         loss = loss_one_hot
#         loss += (6e-3 * loss_var)
#         loss += (1.5e-5 * torch.norm(gen_imgs, 2))      # l2 loss
#         loss += int(args.lambda_s)*loss_distr           # best for noise before BN


#         return loss

#     def validation_step(self, batch, batch_idx):
#         pass

#     def validation_epoch_end(self, outputs):
#         self.student.eval()
#         metric = {}
#         for val_dataloader_name, val_dataloader in self.val_dataloaders.items():
#             for idx, (imgs, labels) in enumerate(val_dataloader):
#                 output = self.student(imgs)
#                 preds = output.max(dim=1)[1]
#                 metric['ce'] += F.cross_entropy(output, labels)
#                 metric['acc'] += (preds == labels).sum() / preds.shape[0]
#             metric['ce'] /= len(val_dataloader)
#             metric['acc'] /= len(val_dataloader)
#             self.log(metric)
 
#     def test_step(self, batch, batch_idx):
#         self.validation_epoch_end()


class FSL(pl.LightningModule):
    def __init__(self, generator_list, teacher, student, optimizer, lr_scheduler, val_dataloaders, gen_ratio):
        super().__init__()
        self.generator_list = generator_list
        self.teacher = teacher
        self.student = student
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.val_dataloaders = val_dataloaders
        self.gen_ratio = gen_ratio
        self.generator_list.eval()
        self.teacher.eval()

    def forward(self, batch):
        imgs, _ = batch
        output = self.student(imgs)
        preds = output.max(dim=1)[1]
        return preds

    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        if torch.rand(1).detach().item() < self.gen_ratio:
            with torch.no_grad():
                imgs, _ = self.generator_list()
        with torch.no_grad():
            outputs_T = self.teacher(imgs)
        outputs_S = self.student(imgs)
        loss = utils.kdloss(outputs_S, outputs_T)
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        self.student.eval()
        metric = {}
        for val_dataloader_name, val_dataloader in self.val_dataloaders.items():
            for idx, (imgs, labels) in enumerate(val_dataloader):
                output = self.student(imgs)
                preds = output.max(dim=1)[1]
                metric['ce'] += F.cross_entropy(output, labels)
                metric['acc'] += (preds == labels).sum() / preds.shape[0]
            metric['ce'] /= len(val_dataloader)
            metric['acc'] /= len(val_dataloader)
            self.log(metric)
 
    def test_step(self, batch, batch_idx):
        self.validation_epoch_end()

    def configure_optimizers(self):
        return self.optimizer
        # return [self.optimizer], [self.lr_scheduler]


def main():
    args = configs.get_args()
    # wandb_logger = pl.loggers.WandbLogger(project="few-short-multi", log_model="all")

    teacher = utils.get_teacher(args)
    student = utils.get_student(args)

    stat_path = args.stat_path + "stats_"+args.dataset+"_"+args.arch+"/stats_"+args.stat_layer+"_"+args.data_type+"_splz"+str(args.stat_bz)+"/"+args.hook_type
    args.num_G = int(len(os.listdir(stat_path))/2)
    generator_list = GeneratorList(args)

    train_dataloader = data.ImagenetDataLoader(imagenet_path=args.imagenet_path).build_trainloader(batch_size = args.batch_size)
    val_dataloader = data.CifarDataLoader(args.dataset).build_testloader()
    optimizer = torch.optim.SGD(student.parameters(), lr=args.lr_S, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = utils.StudentLR(optimizer, args.n_epochs_G, args.decay)

    pipeline = FSL(generator_list, teacher, student, optimizer, lr_scheduler, [val_dataloader], args.ratio) 
    trainer = pl.Trainer(max_steps=10000, log_every_n_steps=10, val_check_interval=500,
                        default_root_dir=args.save_path, # precision=16, # gradient_clip_val=0.5,
                        accelerator="gpu", devices=1, # logger=wandb_logger,
                        callbacks=[pl.callbacks.LearningRateMonitor(logging_interval='step')],
                        # strategy='ddp', num_nodes=args.num_nodes
                        )

    print('start train')
    trainer.fit(pipeline, train_dataloaders=train_dataloader)


if __name__ == "__main__":
    main()

