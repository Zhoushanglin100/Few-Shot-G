export hook_type=$1
export sample_batch=$2
export train_G_bz=$3
export lambda_s=$4
export latent_dim=$5
export partion=$6

export train_S_bz=$7

if [ "$sample_batch" = "8" ]; then 
    export n_divid=7
elif [ "$sample_batch" = "16" ]; then 
    export n_divid=10
elif [ "$sample_batch" = "32" ]; then 
    export n_divid=13
elif [ "$sample_batch" = "64" ]; then 
    export n_divid=8
fi

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



export ext=${n_divid}GC_${hook_type}_R${lambda_s}_ld${latent_dim}_Gbz${train_G_bz}_Sbz${train_S_bz}


if [ "$flag_s1" = "1" ]; then 
    CUDA_VISIBLE_DEVICES=3 python3 gen_stats_cluster_finch_feature.py \
                                        --dataset cifar100 \
                                        --pretrained \
                                        --hook-type $hook_type \
                                        --batch-size $sample_batch \
                                        --ext $ext
fi
if [ "$flag_s2" = "1" ]; then 
    CUDA_VISIBLE_DEVICES=3 python3 DAFLDeepinvert-train_v7_cluster.py \
                                        --dataset cifar100 \
                                        --total_class 100 \
                                        --fix_G \
                                        --train_G \
                                        --n_epochs_G 50 \
                                        --n_epochs 2000 \
                                        --lr_G 0.001 \
                                        --batch_size $train_G_bz \
                                        --n_divid $n_divid \
                                        --hook_type $hook_type \
                                        --lambda_s $lambda_s \
                                        --latent_dim $latent_dim \
                                        --ext $ext
fi
if [ "$flag_s3" = "1" ]; then 
    python3 DAFLDeepinvert-train_v7_cluster.py \
                                        --dataset cifar100 \
                                        --fix_G \
                                        --train_S \
                                        --n_epochs 2000 \
                                        --batch_size $train_S_bz \
                                        --n_divid $n_divid \
                                        --lr_S 0.06 \
                                        --hook_type $hook_type \
                                        --latent_dim $latent_dim \
                                        --resume \
                                        --ext $ext
fi
