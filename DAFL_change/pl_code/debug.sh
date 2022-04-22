export sample_batch=10
export arch_t=resnet34
export arch_s=resnet
export train_G_bz=128
export train_S_bz=256
export lambda_s=10
export latent_dim=3000
export lr_G=0.1
export lr_S=0.01
export ratio=0.5
export imgNet_path=/mnt/data/xingyucai/imagenet/train/
export dataset=cifar10
export ext=Smp${sample_batch}_R${lambda_s}_ld${latent_dim}_Gbz${train_G_bz}_Glr${lr_G}

python3 main.py \
    --dataset $dataset \
    -a $arch_t \
    --arch_s $arch_s \
    --fix_G \
    --train_S \
    --n_epochs 2000 \
    --stat_bz $sample_batch \
    --batch_size $train_S_bz \
    --lr_S $lr_S \
    --latent_dim $latent_dim \
    --ratio $ratio \
    --imagenet_path $imgNet_path \
    --resume \
    --disable_wandb \
    --ext $ext