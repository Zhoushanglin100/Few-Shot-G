WANDB_RUN_ID=bz512-ld2000-eN2000-eG50-lrG0.2-lr0.06-dcy5 python3 DAFLDeepinvert-train_v4.py --dataset cifar10 --channels 3 --n_epochs 2000 --fix_G --batch_size 512 --lr_G 0.2 --lr_S 0.06 --latent_dim 2000 &

WANDB_RUN_ID=bz512-mom-ld2000-eN2000-eG50-lrG0.2-lr0.06-dcy5 python3 DAFLDeepinvert-train_v4.py --dataset cifar10 --channels 3 --n_epochs 2000 --batch_size 512 --lr_G 0.2 --lr_S 0.06 --latent_dim 2000