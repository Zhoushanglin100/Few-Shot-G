# for hook_type in input output
# do
#     for n_divid in 1 5 10 20 40
#     do
#         bash run_cifar100.sh $hook_type $n_divid &
#     done
# done


bash run_cifar100.sh output 20 64 &
bash run_cifar100.sh output 40 64 &

bash run_cifar100.sh output 1 128 &
bash run_cifar100.sh input 1 128