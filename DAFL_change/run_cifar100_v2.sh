
# ------------------------
export hook_type=$1
export n_divid=$2
export train_G_bz=$3
export ext=${n_divid}GC_${hook_type}R10

if n_divid==1
then
    CUDA_VISIBLE_DEVICES=1 python3 gen_stats_cifar_multi.py \
                        --dataset cifar100 \
                        --pretrained \
                        --hook_type $hook_type \
                        --batch-size 32 \
                        --n_divid 1 \
                        --ext $ext
                        
    CUDA_VISIBLE_DEVICES=1 python3 DAFLDeepinvert-train_v6_multi_v6.py \
                                    --dataset cifar100 \
                                    --total_class 100 \
                                    --fix_G \
                                    --train_G \
                                    --n_epochs_G 50 \
                                    --n_epochs 2000 \
                                    --lr_G 0.001 \
                                    --latent_dim 1000 \
                                    --batch_size $train_G_bz \
                                    --n_divid 1 \
                                    --hook_type $hook_type \
                                    --ext $ext

    python3 DAFLDeepinvert-train_v6_multi_v6.py \
                                        --dataset cifar100 \
                                        --total_class 100 \
                                        --fix_G \
                                        --train_S \
                                        --n_epochs 2000 \
                                        --latent_dim 1000 \
                                        --batch_size 64 \
                                        --n_divid 1 \
                                        --lr_S 0.06 \
                                        --hook_type $hook_type \
                                        --resume \
                                        --ext $ext
else
    # python3 gen_stats_cifar_cluster.py \
    #                             --dataset cifar100 \
    #                             --pretrained \
    #                             --hook_type $hook_type \
    #                             --batch-size 32 \
    #                             --n_divid 100 \
    #                             --num_clusters $n_divid \
    #                             --ext $ext

    python3 DAFLDeepinvert-train_v7_cluster.py \
                                --dataset cifar100 \
                                --total_class 100
                                --fix_G \
                                --train_G \
                                --n_epochs_G 50 \
                                --n_epochs 2000 \
                                --lr_G 0.001 \
                                --latent_dim 1000 \
                                --batch_size $train_G_bz \
                                --n_divid $n_divid \
                                --hook_type $hook_type \
                                --ext $ext

    python3 DAFLDeepinvert-train_v7_cluster.py \
                                --dataset cifar100 \
                                --fix_G \
                                --train_S \
                                --n_epochs 2000 \
                                --latent_dim 1000 \
                                --batch_size 64 \
                                --n_divid $n_divid \
                                --lr_S 0.06 \
                                --hook_type $hook_type \
                                --resume \
                                --ext $ext
fi