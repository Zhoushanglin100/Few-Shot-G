# CUDA_VISIBLE_DEVICES=0 python3 DAFLDeepinvert-train_v4.py --dataset cifar10 --channel 3 --batch_size 256 --n_epochs_G 1000 --n_epochs 2000 --lr_G 0.001 --lr_S 0.06 --ext _bz64_G1000_N2000_lrG1e-3_lrS_6e2



CUDA_VISIBLE_DEVICES=2 python3 DAFLDeepinvert-train_v4.py --dataset cifar10 --channels 3 --n_epochs 2000 --fix_G --batch_size 256 --lr_G 0.2 --lr_S 0.06 --latent_dim 1000
