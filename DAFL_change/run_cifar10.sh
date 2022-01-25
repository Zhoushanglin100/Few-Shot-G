export hook_type=$1
export n_divid=$2
export train_G_bz=$3
export lambda_s=$4
export latent_dim=$5
export ext=${n_divid}GC_${hook_type}_ls${lambda_s}_ld${latent_dim}

# CUDA_VISIBLE_DEVICES=1 python3 gen_stats_cluster_finch.py \
#                             --dataset cifar10 \
#                             --pretrained \
#                             --hook-type $hook_type \
#                             --batch-size 16 \
#                             --ext $ext

CUDA_VISIBLE_DEVICES=1 python3 DAFLDeepinvert-train_v7_cluster.py \
                            --dataset cifar10 \
                            --total_class 10 \
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

python3 DAFLDeepinvert-train_v7_cluster.py \
                            --dataset cifar10 \
                            --fix_G \
                            --train_S \
                            --n_epochs 2000 \
                            --batch_size 64 \
                            --n_divid $n_divid \
                            --lr_S 0.06 \
                            --hook_type $hook_type \
                            --latent_dim $latent_dim \
                            --resume \
                            --ext $ext

