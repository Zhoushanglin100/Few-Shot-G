
# bash run_local.sh a s1 10 resnet34 resnet 128 128 10 1000 0.1 0.01 0.5 /data/imagenet/
# bash run_local.sh a s1 50 resnet34 resnet 128 128 10 1000 0.1 0.01 0.5 /data/imagenet/
# bash run_local.sh a s1 100 resnet34 resnet 128 128 10 1000 0.1 0.01 0.5 /data/imagenet/

# bash run_local.sh a s1 10 vgg16 vgg 128 128 10 1000 0.1 0.01 0.5 /data/imagenet/
# bash run_local.sh a s1 50 vgg16 vgg 128 128 10 1000 0.1 0.01 0.5 /data/imagenet/
# bash run_local.sh a s1 100 vgg16 vgg 128 128 10 1000 0.1 0.01 0.5 /data/imagenet/

# for sample_bz in 10 50 100
# do
#     for teacher in resnet34 vgg16
#     do
#         bash run_local.sh a s1 $sample_bz $teacher resnet 128 128 10 1000 0.1 0.01 0.5 /data/imagenet/ &
#     done
# done

# -----------------
for G_bz in 128 256
do
    for latent_dim in 1000 3000 5000
    do
    bash run_local.sh a s2 10 resnet34 resnet $G_bz 128 10 $latent_dim 0.1 0.01 0.1 /data/imagenet/ &
    bash run_local.sh a s2 50 resnet34 resnet $G_bz 128 10 $latent_dim 0.1 0.01 0.1 /data/imagenet/ &
    bash run_local.sh a s2 100 resnet34 resnet $G_bz 128 10 $latent_dim 0.1 0.01 0.1 /data/imagenet/ &
    bash run_local.sh a s2 10 vgg16 vgg $G_bz 128 10 $latent_dim 0.1 0.01 0.1 /data/imagenet/ &
    bash run_local.sh a s2 50 vgg16 vgg $G_bz 128 10 $latent_dim 0.1 0.01 0.1 /data/imagenet/ &
    bash run_local.sh a s2 100 vgg16 vgg $G_bz 128 10 $latent_dim 0.1 0.01 0.1 /data/imagenet/ &
    done
done

# -----------------

# bash run_local.sh a s3 10 resnet34 resnet $G_bz $S_bz 10 $latent_dim 0.1 0.01 $ratio /data/imagenet/
# bash run_local.sh a s3 50 resnet34 resnet $G_bz $S_bz 10 $latent_dim 0.1 0.01 $ratio /data/imagenet/
# bash run_local.sh a s3 100 resnet34 resnet $G_bz $S_bz 10 $latent_dim 0.1 0.01 $ratio /data/imagenet/

# bash run_local.sh a s3 10 vgg16 vgg $G_bz $S_bz 10 $latent_dim 0.1 0.01 $ratio /data/imagenet/
# bash run_local.sh a s3 50 vgg16 vgg $G_bz $S_bz 10 $latent_dim 0.1 0.01 $ratio /data/imagenet/
# bash run_local.sh a s3 100 vgg16 vgg $G_bz $S_bz 10 $latent_dim 0.1 0.01 $ratio /data/imagenet/

# bash run_local.sh a s3 10 vgg16 resnet $G_bz $S_bz 10 $latent_dim 0.1 0.01 $ratio /data/imagenet/
# bash run_local.sh a s3 50 vgg16 resnet $G_bz $S_bz 10 $latent_dim 0.1 0.01 $ratio /data/imagenet/
# bash run_local.sh a s3 100 vgg16 resnet $G_bz $S_bz 10 $latent_dim 0.1 0.01 $ratio /data/imagenet/
