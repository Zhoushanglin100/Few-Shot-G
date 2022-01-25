# for hook_type in output input
# do
#     for n_divid in 1 2 5 8 10 20 40
#     do
#         for lambda_s in 1 3 10 50 100
#         do
#             for latent_dim in 1000 2000 3000
#             do
#                 bash run_cifar100.sh $hook_type $n_divid 64 $lambda_s $latent_dim &
#             done
#         done
#     done
# done


# bash run_cifar100.sh output 20 64 &
# bash run_cifar100.sh output 40 64 &
# bash run_cifar100.sh output 1 128 &
# bash run_cifar100.sh input 1 128



# --------------------------------------------
bash run_cifar100_v4.sh output 8 64 100 1000 $1 &
bash run_cifar100_v4.sh output 8 64 100 3000 $1 &

bash run_cifar100_v4.sh output 8 64 10 1000 $1 &
bash run_cifar100_v4.sh output 8 64 10 3000 $1 &

# --------------------------------------------
bash run_cifar100_v4.sh output32 64 100 1000 $1 &
bash run_cifar100_v4.sh output32 64 100 3000 $1 &

bash run_cifar100_v4.sh output32 64 10 1000 $1 &
bash run_cifar100_v4.sh output32 64 10 3000 $1 &

