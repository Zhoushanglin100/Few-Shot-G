export hook_type=$1
export n_divid=$2
export train_G_bz=$3
export ext=${n_divid}GC_${hook_type}R10

python3 gen_stats_cifar_multi.py \
                        --pretrained \
                        --hook_type input \
                        --batch-size 32 \
                        --n_divid 1 \
                        --ext 5GF_InR100


python3 DAFLDeepinvert-train_v6_multi_v6.py \
                                  --fix_G \
                                  --train_G \
                                  --n_epochs_G 50 \
                                  --n_epochs 2000 \
                                  --lr_G 0.001 \
                                  --latent_dim 1000 \
                                  --batch_size 128 \
                                  --n_divid 1 \
                                  --hook_type input \
                                  --ext 5GF_InR100

python3 DAFLDeepinvert-train_v6_multi_v6.py \
                                    --fix_G \
                                    --train_S \
                                    --n_epochs 2000 \
                                    --latent_dim 1000 \
                                    --batch_size 64 \
                                    --n_divid 1 \
                                    --lr_S 0.06 \
                                    --hook_type input \
                                    --resume \
                                    --ext 5GF_InR100