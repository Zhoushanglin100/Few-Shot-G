import os
import torch
# from script import *

import torch.nn as nn

import configs
import models
import utils
import trainer
import data

def main():

    # set up
    args = configs.get_args()
    log_func = configs.set_logger(args)

    # get teacher
    teacher = utils.get_teacher(args)
    teacher = nn.DataParallel(teacher.cuda())
    
    # -------------------------------------
    global acc, acc_best
    acc = 0
    acc_best = 0
    start_epoch = 1

    stat_path = args.stat_path + "stats_"+args.dataset+"_"+args.arch+"/stats_"+args.stat_layer+"_"+args.data_type+"_splz"+str(args.stat_bz)+"/"+args.hook_type

    # ------------------------------------------------

    if args.train_G:
        # setting up the range for jitter
        lim_0, lim_1 = 2, 2
        n = int(len(os.listdir(stat_path))/2)
        print("!!!!!!!!", n, "!!!!!!!!!!!!!")

        ### iteratively train generators
        for idx in range(0, n):
            
            start_class = idx
            end_class = idx+1
            print("\n !!!!! start_class: "+str(start_class)+" end_class: "+str(end_class))

            ### way to resume generator
            save_name = "start-"+str(start_class)+"_end-"+str(end_class)+".pth"
            if os.path.exists(args.save_path+"/"+save_name):
                ckeckpoints = torch.load(args.save_path+"/"+save_name)
                if ckeckpoints["epoch"] == args.n_epochs_G:
                    print("Generate exits!!", save_name)
                    continue
            
            generator = models.Generator(latent_dim=args.latent_dim, img_size=args.img_size).cuda()
            generator = nn.DataParallel(generator)

            ### optimization
            criterion = torch.nn.CrossEntropyLoss().cuda()
            optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr_G)
            scheduler_G = utils.GeneratorLR(optimizer_G)

            ### load student
            net = utils.get_student(args)
            net.cuda()
            # optimizer_S = torch.optim.SGD(net.parameters(), lr=args.lr_S, momentum=0.9, weight_decay=5e-4)

            ### Create hooks for feature statistics catching
            mean_layers_dictionary = torch.load(stat_path+"/mean_"+args.arch+"_start-"+str(start_class)+"_end-"+str(end_class)+".pth")
            var_layers_dictionary = torch.load(stat_path+"/var_"+args.arch+"_start-"+str(start_class)+"_end-"+str(end_class)+".pth")

            mean_layers_dictionary = {f'module.{k}': v for k, v in mean_layers_dictionary.items()}
            var_layers_dictionary = {f'module.{k}': v for k, v in var_layers_dictionary.items()}

            loss_r_feature_layers = []
            name_layer = []
            for name, module in teacher.named_modules():
                if name in mean_layers_dictionary.keys():
                    layers = models.DeepInversionFeatureHook(args.hook_type, args.stat_type, name, module, mean_layers_dictionary, var_layers_dictionary)
                    loss_r_feature_layers.append(layers)
                    name_layer.append(name)

            ### start training generator
            for e in range(start_epoch, args.n_epochs_G+1):

                trainer.train_G(args, idx, net, generator, teacher, e, 
                                optimizer_G, scheduler_G, criterion, 
                                lim_0, lim_1,
                                loss_r_feature_layers)
            
                torch.save({'epoch': e,
                            'G_state_dict': generator.module.state_dict(),
                            'G_optimizer_state_dict':optimizer_G.state_dict()}, 
                            args.save_path+"/"+save_name)

            # ------------------------------------------------
            for layers in loss_r_feature_layers:
                layers.close()
            torch.cuda.empty_cache()

    # ------------------------------------------------
    ### train student
    if args.train_S:
        start_epoch = args.n_epochs_G

        print("Stats path:", stat_path)
        num_G = int(len(os.listdir(stat_path))/2)

        ### load student and generator
        gen_info = utils.generator_info(num_G, args.save_path, 
                                         args.batch_size, args.latent_dim, 
                                         args.img_size, args.channels)
        # G_list = gen_info.get_generators()        
        gen_info.get_generators()
        student = utils.get_student(args).cuda()

        ### load training data
        imagenet_dataloader = data.ImagenetDataLoader().build_trainloader(batch_size = args.batch_size)
        eval_dataloader = data.CifarDataLoader(args.dataset).build_testloader()
        
        ### optimization
        criterion = torch.nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(student.parameters(), lr=args.lr_S, momentum=0.9, weight_decay=5e-4)
        scheduler = utils.StudentLR(optimizer, args.n_epochs_G, args.decay)
        # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer_S, start_factor=args.lr_S, total_iters=int(args.n_epochs-args.n_epochs_G))

        save_name = "imgNet_{}{}_{}_trainS_ld{}_eN{}_lrS{}_dcy{}_lambda{}.pth".format(args.dataset, 
                                                                                        args.arch_s,
                                                                                        args.ratio, 
                                                                                        args.latent_dim,
                                                                                        args.n_epochs,
                                                                                        args.lr_S,
                                                                                        args.decay, 
                                                                                        args.lambda_s)
        save_path = args.save_path + '/' + save_name

        if args.resume and os.path.exists(save_path):
            print("!!!! RESUME !!!!")
            start_epoch, student, optimizer, scheduler = utils.load(save_path, student, optimizer, scheduler)

        student = nn.DataParallel(student)
        for epoch in range(start_epoch, args.n_epochs):
            log_func({"epoch": epoch})
            trainer.train_S(student, teacher, epoch, optimizer, scheduler, imagenet_dataloader, gen_info, 
                             args.ratio, fix_G=args.fix_G, log_func=log_func)
            utils.save(epoch, student.module, optimizer, scheduler, save_path)
            trainer.eval(student, eval_dataloader, criterion, log_func=log_func)

#############################################################
if __name__ == '__main__':
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    # torch.multiprocessing.set_start_method('spawn', force=True)
    main()
