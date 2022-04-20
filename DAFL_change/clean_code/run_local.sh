
# CUDA_VISIBLE_DEVICES=5 python3 gen_stats_cluster_finch_feature.py --pretrained --dataset cifar10 -a resnet34 --data-type sample --batch-size 10 --stat-layer all

# CUDA_VISIBLE_DEVICES=5 python3 DAFLDeepinvert-train_v9.py --dataset cifar10 -a resnet34 --fix_G --train_G --n_epochs_G 100 --lr_G 0.1 --latent_dim 1000 --stat_bz 10 --stat_layer all --ext xxx

# python3 DAFLDeepinvert-train_v7_cluster.py --dataset cifar10 --fix_G --train_S --n_epochs 2000 -a resnet34 --arch_s resnet --batch_size 128 --lr_S 0.01 --stat_bz 10 --stat_layer all --latent_dim 1000 --resume --ext xxx

## bash run_v9_local.sh a s1 10 resnet34 resnet 128 128 10 1000 0.1 0.01 0.5 /data/imagenet/

export partion=$1
export step=$2
if [ "$step" = "s1" ]; then 
    export flag_s1=1
    export flag_s2=0
    export flag_s3=0
elif [ "$step" = "s2" ]; then 
    export flag_s1=0
    export flag_s2=1
    export flag_s3=0
elif [ "$step" = "s3" ]; then 
    export flag_s1=0
    export flag_s2=0
    export flag_s3=1
fi

export sample_batch=$3

export arch_t=$4
export arch_s=$5

export train_G_bz=$6
export train_S_bz=$7

export lambda_s=$8
export latent_dim=$9

export lr_G=${10}
export lr_S=${11}

export ratio=${12}
export imgNet_path=${13}

export dataset=cifar10

export ext=Smp${sample_batch}_R${lambda_s}_ld${latent_dim}_Gbz${train_G_bz}_Glr${lr_G}



if [ "$flag_s1" = "1" ]; then 
    CUDA_VISIBLE_DEVICES=5 python3 gen_stats_cluster_finch_feature.py \
                                        --dataset $dataset \
                                        -a $arch_t \
                                        --pretrained \
                                        --batch-size $sample_batch
fi
if [ "$flag_s2" = "1" ]; then 
    numG=$(python3 findN.py -a $arch_t --stat_bz $sample_batch)
    for idx in $(seq 0 $numG)
    do
        CUDA_VISIBLE_DEVICES=5 python3 main_sepG.py \
                                            --dataset $dataset \
                                            -a $arch_t \
                                            --fix_G \
                                            --train_G \
                                            --stat_bz $sample_batch \
                                            --batch_size $train_G_bz \
                                            --n_epochs_G 50 \
                                            --lr_G $lr_G \
                                            --lambda_s $lambda_s \
                                            --latent_dim $latent_dim \
                                            --Gindex $idx \
                                            --disable_wandb \
                                            --ext $ext &
    done
fi
if [ "$flag_s3" = "1" ]; then 
    CUDA_VISIBLE_DEVICES=4,5 python3 main.py \
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
                                --ext $ext
fi
