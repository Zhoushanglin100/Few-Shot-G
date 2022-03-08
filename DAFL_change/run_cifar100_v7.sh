export hook_type=$1
export sample_batch=$2
export train_G_bz=$3
export lambda_s=$4
export latent_dim=$5
export train_S_bz=$6

export data_type=$7

export step=$8
if [ "$step" = "s1" ]; then 
    export flag_s1=1
    export flag_s2=1
    export flag_s3=1
elif [ "$step" = "s2" ]; then 
    export flag_s1=0
    export flag_s2=1
    export flag_s3=1
elif [ "$step" = "s3" ]; then 
    export flag_s1=0
    export flag_s2=0
    export flag_s3=1
fi

if [ "$data_type" = "sample" ]; then
    export ext=NC_Smp${sample_batch}_${hook_type}_R${lambda_s}_ld${latent_dim}_Gbz${train_G_bz}_Sbz${train_S_bz}
elif [ "$data_type" = "sample" ]; then
    export ext=${data_type}_Smp${sample_batch}_${hook_type}_R${lambda_s}_ld${latent_dim}_Gbz${train_G_bz}_Sbz${train_S_bz}
fi

export partion=$9


if [ "$flag_s1" = "1" ]; then 
    srun -p $partion --gres=gpu:1 -n 1 --cpus-per-task=4 --exclude=asimov-157 python3 gen_stats_cluster_finch_feature.py \
                                        --dataset cifar100 \
                                        --data-type $data_type \
                                        --pretrained \
                                        --thrd 20 \
                                        --hook-type $hook_type \
                                        --batch-size $sample_batch \
                                        --ext $ext
fi
if [ "$flag_s2" = "1" ]; then 
    srun -p $partion --gres=gpu:1 -n 1 --cpus-per-task=4 --exclude=asimov-157 python3 DAFLDeepinvert-train_v7_cluster.py \
                                        --dataset cifar100 \
                                        --fix_G \
                                        --train_G \
                                        --n_epochs_G 50 \
                                        --lr_G 0.001 \
                                        --stat_bz $sample_batch \
                                        --batch_size $train_G_bz \
                                        --hook_type $hook_type \
                                        --lambda_s $lambda_s \
                                        --latent_dim $latent_dim \
                                        --ext $ext
fi
if [ "$flag_s3" = "1" ]; then 
    srun -p $partion --gres=gpu:4 -n 1 --cpus-per-task=16 --exclude=asimov-157 python3 DAFLDeepinvert-train_v7_cluster.py \
                                        --dataset cifar100 \
                                        --fix_G \
                                        --train_S \
                                        --n_epochs 2000 \
                                        --stat_bz $sample_batch \
                                        --batch_size $train_S_bz \
                                        --lr_S 0.06 \
                                        --hook_type $hook_type \
                                        --latent_dim $latent_dim \
                                        --resume \
                                        --ext $ext
fi
