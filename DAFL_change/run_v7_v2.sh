for hook_type in output input
do
    for n_divid in 1 2 5 8 10 20 40
    do
        for latent_dim in 1000 2000 3000
        do
            bash run_cifar100.sh $hook_type $n_divid 64 $lambda_s $latent_dim &
        done
    done
done


# bash run_cifar100.sh output 20 64 &
# bash run_cifar100.sh output 40 64 &
# bash run_cifar100.sh output 1 128 &
# bash run_cifar100.sh input 1 128