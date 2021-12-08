CUDA_VISIBLE_DEVICES=1 python3 DAFLDeepinvert-train_v6_multi_v3.py \
                                    --fix_G \
                                    --train_G \
                                    --n_epochs 2000 \
                                    --lr_G 0.001 \
                                    --latent_dim 1000 \
                                    --batch_size 256 \
                                    --n_divid 1 \
                                    --ext 1Ghook

# CUDA_VISIBLE_DEVICES=1 python3 DAFLDeepinvert-train_v6_multi_v3.py \
#                                     --fix_G \
#                                     --train_S \
#                                     --n_epochs 2000 \
#                                     --latent_dim 1000 \
#                                     --batch_size 256 \
#                                     --n_divid 1 \
#                                     --lr_S 0.06 \
#                                     --num_sample 100 \
#                                     --ext 1Ghookv2 