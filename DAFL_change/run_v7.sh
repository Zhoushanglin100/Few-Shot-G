# #### 5 clusters
# ### input as stats
# ### step 1: generate statistics for each class
# python3 gen_stats_cifar_cluster.py \
#                         --pretrained \
#                         --hook_type input \
#                         --batch-size 32 \
#                         --n_divid 10 \
#                         --num_clusters 5 \
#                         --ext 5GC_InR100

# ### step 2: train multiple generators
# python3 DAFLDeepinvert-train_v7_cluster.py \
#                                   --fix_G \
#                                   --train_G \
#                                   --n_epochs_G 50 \
#                                   --n_epochs 2000 \
#                                   --lr_G 0.001 \
#                                   --latent_dim 1000 \
#                                   --batch_size 128 \
#                                   --n_divid 5 \
#                                   --hook_type input \
#                                   --ext 5GC_InR100

# ### step 3: train student
# python3 DAFLDeepinvert-train_v7_cluster.py \
#                                     --fix_G \
#                                     --train_S \
#                                     --n_epochs 2000 \
#                                     --latent_dim 1000 \
#                                     --batch_size 64 \
#                                     --n_divid 5 \
#                                     --lr_S 0.06 \
#                                     --hook_type input \
#                                     --resume \
#                                     --ext 5GC_InR100


# # -------------------------------------------
# #### 5 clusters
# ### output as stats
# ### step 1: generate statistics for each class
# python3 gen_stats_cifar_cluster.py \
#                         --pretrained \
#                         --hook_type output \
#                         --batch-size 32 \
#                         --n_divid 10 \
#                         --num_clusters 5 \
#                         --ext 5GC_OutR100

# ### step 2: train multiple generators
# python3 DAFLDeepinvert-train_v7_cluster.py \
#                                   --fix_G \
#                                   --train_G \
#                                   --n_epochs_G 50 \
#                                   --n_epochs 2000 \
#                                   --lr_G 0.001 \
#                                   --latent_dim 1000 \
#                                   --batch_size 128 \
#                                   --n_divid 5 \
#                                   --hook_type output \
#                                   --ext 5GC_OutR100

# ### step 3: train student
# python3 DAFLDeepinvert-train_v7_cluster.py \
#                                     --fix_G \
#                                     --train_S \
#                                     --n_epochs 2000 \
#                                     --latent_dim 1000 \
#                                     --batch_size 64 \
#                                     --n_divid 5 \
#                                     --lr_S 0.06 \
#                                     --hook_type output \
#                                     --resume \
#                                     --ext 5GC_OutR100

# #############################################################
#### 3 clusters
### input as stats
### step 1: generate statistics for each class
python3 gen_stats_cifar_cluster.py \
                        --pretrained \
                        --hook_type input \
                        --batch-size 32 \
                        --n_divid 10 \
                        --num_clusters 3 \
                        --ext 3GC_InR100

### step 2: train multiple generators
python3 DAFLDeepinvert-train_v7_cluster.py \
                                  --fix_G \
                                  --train_G \
                                  --n_epochs_G 50 \
                                  --n_epochs 2000 \
                                  --lr_G 0.001 \
                                  --latent_dim 1000 \
                                  --batch_size 128 \
                                  --n_divid 3 \
                                  --hook_type input \
                                  --ext 3GC_InR100

### step 3: train student
python3 DAFLDeepinvert-train_v7_cluster.py \
                                    --fix_G \
                                    --train_S \
                                    --n_epochs 2000 \
                                    --latent_dim 1000 \
                                    --batch_size 64 \
                                    --n_divid 3 \
                                    --lr_S 0.06 \
                                    --hook_type input \
                                    --resume \
                                    --ext 3GC_InR100


# -------------------------------------------
#### 3 clusters
### output as stats
### step 1: generate statistics for each class
python3 gen_stats_cifar_cluster.py \
                        --pretrained \
                        --hook_type output \
                        --batch-size 32 \
                        --n_divid 10 \
                        --num_clusters 3 \
                        --ext 3GC_OutR100

### step 2: train multiple generators
python3 DAFLDeepinvert-train_v7_cluster.py \
                                  --fix_G \
                                  --train_G \
                                  --n_epochs_G 50 \
                                  --n_epochs 2000 \
                                  --lr_G 0.001 \
                                  --latent_dim 1000 \
                                  --batch_size 128 \
                                  --n_divid 3 \
                                  --hook_type output \
                                  --ext 3GC_OutR100

### step 3: train student
python3 DAFLDeepinvert-train_v7_cluster.py \
                                    --fix_G \
                                    --train_S \
                                    --n_epochs 2000 \
                                    --latent_dim 1000 \
                                    --batch_size 64 \
                                    --n_divid 3 \
                                    --lr_S 0.06 \
                                    --hook_type output \
                                    --resume \
                                    --ext 3GC_OutR100