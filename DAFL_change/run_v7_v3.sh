### running
# bash run_cifar100_v4.sh input 32 64 10 3000 $1 s3 &
# bash run_cifar100_v4.sh input 8 64 10 1000 $1 &
# bash run_cifar100_v4.sh input 8 64 10 3000 $1 &
# bash run_cifar100_v4.sh output 8 64 100 1000 $1 &

### crashed
bash run_cifar100_v4.sh output 8 64 10 3000 $1 s3 &
bash run_cifar100_v4.sh output 8 64 10 1000 $1 s3 &

bash run_cifar100_v4.sh output 8 64 100 3000 $1 s3 &

#
bash run_cifar100_v4.sh input 8 64 100 3000 $1 s3 &
bash run_cifar100_v4.sh input 8 64 100 1000 $1 s3 &

bash run_cifar100_v4.sh input 32 64 10 1000 $1 s3 &
bash run_cifar100_v4.sh input 32 64 100 1000 $1 s3 &

bash run_cifar100_v4.sh input 32 64 100 3000 $1 s3 &

### failed

bash run_cifar100_v4.sh output 32 64 100 1000 $1 s2 &
bash run_cifar100_v4.sh output 32 64 100 3000 $1 s2 &

bash run_cifar100_v4.sh output 32 64 10 1000 $1 s2 &
bash run_cifar100_v4.sh output 32 64 10 3000 $1 s2 &

