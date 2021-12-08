# python3 DAFLDeepinvert-train_v6_tmp_v2.py --fix_G --train_G --n_epochs 2000 --lr_G 0.001 --latent_dim 1000 --ext 10G1000
# python3 DAFLDeepinvert-train_v6_tmp_v2.py --fix_G --train_S --n_epochs 2000 --latent_dim 1000 --ext 10G1000

# python3 DAFLDeepinvert-train_v6_tmp_v2.py --fix_G --train_G --n_epochs 2000 --lr_G 0.001 --latent_dim 1000 --ext 10G1000
# python3 DAFLDeepinvert-train_v6_tmp_v2.py --fix_G --train_S --n_epochs 2000 --latent_dim 3000 --ext 10G3000NEW


# CUDA_VISIBLE_DEVICES=3 python3 DAFLDeepinvert-train_v6_multi_v4.py \
#                                   --fix_G \
#                                   --train_G \
#                                   --n_epochs_G 50 \
#                                   --n_epochs 2000 \
#                                   --lr_G 0.001 \
#                                   --latent_dim 1000 \
#                                   --batch_size 256 \
#                                   --n_divid 1 \
#                                   --num_sample 100 \
#                                   --hook_type input \
#                                   --ext 1Gextract_InR100

CUDA_VISIBLE_DEVICES=1 python3 DAFLDeepinvert-train_v6_multi_v4.py \
                                    --fix_G \
                                    --train_S \
                                    --n_epochs 2000 \
                                    --latent_dim 1000 \
                                    --batch_size 256 \
                                    --n_divid 1 \
                                    --lr_S 0.06 \
                                    --num_sample 100 \
                                    --hook_type output \
                                    --resume \
                                    --ext 1GextractR100
