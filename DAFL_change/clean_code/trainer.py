import random
import torch
from torch.autograd import Variable
import utils
import configs

args = configs.get_args()
log_func = configs.set_logger(args)

def train_G(args, idx, net, generator, teacher, epoch,
            optimizer_G, scheduler_G, criterion, 
            lim_0, lim_1,
            loss_r_feature_layers): 
    
    log_func({"epoch": epoch})

    net.train()
    loss = None

    num_itr = 500
    for i in range(num_itr):

        z = Variable(torch.randn(args.batch_size, args.latent_dim)).cuda()
        gen_imgs = generator(z)

        ### one-hot loss
        outputs_T = teacher(gen_imgs)
        pred = outputs_T.data.max(1)[1]
        loss_one_hot = criterion(outputs_T,pred)
        # softmax_o_T = torch.nn.functional.softmax(outputs_T, dim = 1).mean(dim = 0)
        # loss_information_entropy = (softmax_o_T * torch.log10(softmax_o_T)).sum()
        # loss_activation = -features_T.abs().mean()
        # loss = loss_one_hot * args.oh + loss_information_entropy * args.ie + loss_activation * args.a 
        
        ### KD loss
        outputs_S = net(gen_imgs)
        loss_kd = utils.kdloss(outputs_S, outputs_T)

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
        loss += (1.5e-5 * torch.norm(gen_imgs, 2))      # l2 loss
        loss += int(args.lambda_s)*loss_distr           # best for noise before BN

        if i % 100 == 0:
            print('Train G_%d, Epoch %d, Batch: %d, Loss: %f' % (idx, epoch, i, loss.data.item()))
            loss_dict = {"loss_G/OneHot_Loss_"+str(idx): loss_one_hot.item(), 
                            "loss_G/KD_Loss_S_"+str(idx): loss_kd.item(),
                            "loss_G/Var_Loss_"+str(idx): loss_var.item(),
                            "loss_G/R_Loss_"+str(idx): loss_distr,
                            "loss_G/L2_Loss_"+str(idx): torch.norm(gen_imgs, 2).item(),
                            "G_perf/total_loss_"+str(idx): loss.data.item()
                        }
            log_func(loss_dict, step=(i+epoch*num_itr)+idx*num_itr*args.n_epochs_G)
            log_func({"lr/lr_G": optimizer_G.param_groups[0]['lr']})
        
        optimizer_G.zero_grad()
        loss.backward()
        optimizer_G.step()
    scheduler_G.step()


def train_S(student, teacher, epoch, optimizer, scheduler, dataloader, gen_info, gen_ratio, fix_G, log_func=print):
    log_func({"epoch": epoch})
    # print("\n>>>>> Train Student <<<<<")
    student.train()
    num_itr = 500
    for batch_idx, (imgs, _) in enumerate(dataloader):
        if batch_idx > num_itr:
            break

        if torch.rand(1).item() < gen_ratio:
            # imgs, _ = utils.generate_imgs(G_list, batch_size, latent_dim)
            imgs, _ = gen_info.generate_imgs()
        imgs = imgs.cuda()

        # batch_size = imgs.shape[0]
        # print("aaaaaaa", batch_size)
        
        optimizer.zero_grad()
        outputs_T= teacher(imgs)
        outputs_S = student(imgs)
        loss_kd = utils.kdloss(outputs_S, outputs_T)
        
        ### only train student after n_epochs_G epochs
        if fix_G:
            loss = loss_kd
        else:
            raise NotImplementedError
            # loss = loss + (6e-3 * loss_var)
            # loss = loss + (1.5e-5 * torch.norm(gen_imgs, 2))      # l2 loss
            # loss = loss + loss_distr                              # best for noise before BN
            # loss = loss + loss_kd
        if batch_idx % 100 == 0:
            print('Student Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, batch_idx, loss.data.item()))

        log_func({"total_loss_S": loss.item()})
        log_func({"lr/lr_S": optimizer.param_groups[0]['lr']})

        loss.backward()
        optimizer.step()
    scheduler.step()


def eval(net, dataloader, criterion, log_func=print):
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    total_len = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            output = net(images)
            avg_loss += criterion(output, labels).sum()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
            total_len += labels.size(0)
    avg_loss /= len(dataloader)
    acc = float(total_correct) / total_len
    print('\n|||| Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), acc))
    log_func({"test_loss": avg_loss.data.item(), "test_acc": acc})

