### step 1: generate statistics for each class
python3 gen_stats_cifar_multi.py --pretrained --hook_type input

### step 2: train multiple generators
python3 DAFLDeepinvert-train_v6_multi_v5.py \
                                  --fix_G \
                                  --train_G \
                                  --n_epochs_G 50 \
                                  --n_epochs 2000 \
                                  --lr_G 0.001 \
                                  --latent_dim 1000 \
                                  --batch_size 256 \
                                  --n_divid 10 \
                                  --hook_type input \
                                  --ext 10G_InR100

### step 3: train student
python3 DAFLDeepinvert-train_v6_multi_v5.py \
                                    --fix_G \
                                    --train_S \
                                    --n_epochs 2000 \
                                    --latent_dim 1000 \
                                    --batch_size 256 \
                                    --n_divid 10 \
                                    --lr_S 0.06 \
                                    --hook_type input \
                                    --resume \
                                    --ext 10G_InR100


# ---------------------------------------------------------------


python3 gen_stats_cifar_multi.py --pretrained --hook_type output

python3 DAFLDeepinvert-train_v6_multi_v5.py \
                                    --fix_G \
                                    --train_S \
                                    --n_epochs 2000 \
                                    --latent_dim 1000 \
                                    --batch_size 256 \
                                    --n_divid 10 \
                                    --lr_S 0.06 \
                                    --hook_type output \
                                    --resume \
                                    --ext 10G_OutR100

python3 DAFLDeepinvert-train_v6_multi_v5.py \
                                  --fix_G \
                                  --train_G \
                                  --n_epochs_G 50 \
                                  --n_epochs 2000 \
                                  --lr_G 0.001 \
                                  --latent_dim 1000 \
                                  --batch_size 256 \
                                  --n_divid 10 \
                                  --hook_type output \
                                  --ext 10G_OutR100