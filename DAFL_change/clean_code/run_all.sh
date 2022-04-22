export partion=$1
export imagenet_path=$2

# # -----------------

# for sample_bz in 10 50 100
# do
#     for teacher in resnet34 vgg16
#     do
#         bash run_hpc.sh $partion s1 $sample_bz $teacher resnet 128 128 10 1000 0.1 0.01 0.5 $imagenet_path &
#     done
# done

# # -----------------
# for G_bz in 128 256
# do
#     for latent_dim in 1000 3000 5000
#     do
#         bash run_hpc.sh $partion s2 10 vgg16 vgg $G_bz 128 10 $latent_dim 0.1 0.01 0.1 $imagenet_path cifar10 &
#         bash run_hpc.sh $partion s2 50 resnet34 resnet $G_bz 128 10 $latent_dim 0.1 0.01 0.1 $imagenet_path cifar10 &
#         bash run_hpc.sh $partion s2 100 resnet34 resnet $G_bz 128 10 $latent_dim 0.1 0.01 0.1 $imagenet_path cifar10 &
        
#         bash run_hpc.sh $partion s2 10 resnet34 resnet $G_bz 128 10 $latent_dim 0.1 0.01 0.1 $imagenet_path cifar10 &
#         bash run_hpc.sh $partion s2 50 vgg16 vgg $G_bz 128 10 $latent_dim 0.1 0.01 0.1 $imagenet_path cifar10 &
#         bash run_hpc.sh $partion s2 100 vgg16 vgg $G_bz 128 10 $latent_dim 0.1 0.01 0.1 $imagenet_path cifar10 &
#     done
# done


# -----------------

for G_bz in 128 256; do
for S_bz in 64 128 256; do
for latent_dim in 100 1000 3000 5000; do
for ratio in $(seq 0 0.1 1); do
for $fewS in 10 50 100
bash run_hpc.sh $partion s3 $fewS resnet34 resnet $G_bz $S_bz 10 $latent_dim 0.1 0.01 $ratio $imagenet_path cifar10 &
bash run_hpc.sh $partion s3 $fewS vgg16 vgg $G_bz $S_bz 10 $latent_dim 0.1 0.01 $ratio $imagenet_path cifar10 &
bash run_hpc.sh $partion s3 $fewS vgg16 resnet $G_bz $S_bz 10 $latent_dim 0.1 0.01 $ratio $imagenet_path cifar10 &

done
done
done
done