for hook_type in input output
do
    for n_divid in 1 5 10 20 40
    do
        bash run_cifar100.sh $hook_type $n_divid
    done
done