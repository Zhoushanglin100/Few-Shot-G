# ### finish
# bash run_cifar100_v4.sh output 8 64 100 3000 $1 s3 &
# bash run_cifar100_v4.sh output 8 64 100 1000 $1 s3 &
# bash run_cifar100_v4.sh output 8 64 10 3000 $1 s3 &
# bash run_cifar100_v4.sh output 8 64 10 1000 $1 s3 &

# bash run_cifar100_v4.sh input 8 64 10 1000 $1 s3 &
# bash run_cifar100_v4.sh input 8 64 10 3000 $1 s3 &
# bash run_cifar100_v4.sh input 8 64 100 1000 $1 s3 &
# bash run_cifar100_v4.sh input 8 64 100 3000 $1 s3 &
# bash run_cifar100_v4.sh input 32 64 100 1000 $1 s3 &


# ### runing
# bash run_cifar100_v4.sh output 32 64 100 1000 $1 s3 &
# bash run_cifar100_v4.sh output 32 64 100 3000 $1 s3 &

# ### crashed
# bash run_cifar100_v4.sh output 32 64 10 1000 $1 s3 &
# bash run_cifar100_v4.sh output 32 64 10 3000 $1 s3 &

# #
# bash run_cifar100_v4.sh input 32 64 10 1000 $1 s3 &
# bash run_cifar100_v4.sh input 32 64 100 3000 $1 s3 &
# bash run_cifar100_v4.sh input 32 64 10 3000 $1 s3 &

# #
# bash run_cifar100_v5.sh output 16 128 10 3000 $1 128 s3 &
# bash run_cifar100_v5.sh output 16 128 100 3000 $1 128 s3 &

# bash run_cifar100_v5.sh input 16 128 10 3000 $1 128 s3 &
# bash run_cifar100_v5.sh input 16 128 100 3000 $1 128 s3 &


# -------------------------------

# bash run_cifar100_v6.sh output 2 128 100 3000 128 s3 $1 &
# bash run_cifar100_v6.sh output 2 128 100 1000 128 s3 $1 &
# bash run_cifar100_v6.sh output 2 128 10 1000 128 s3 $1 &
# bash run_cifar100_v6.sh output 2 128 10 3000 128 s3 $1 &

bash run_cifar100_v6.sh output 1 128 100 3000 128 s3 $1 &
# bash run_cifar100_v6.sh output 1 128 100 1000 128 s3 $1 &
bash run_cifar100_v6.sh output 1 128 10 3000 128 s3 $1 &
bash run_cifar100_v6.sh output 1 128 10 1000 128 s3 $1 &

### 
# bash run_cifar10_v2.sh output 2 128 100 3000 128 s3 $1 &
# bash run_cifar10_v2.sh output 1 128 100 3000 128 s3 $1 &
# bash run_cifar10_v2.sh output 2 128 10 3000 128 s3 $1 &
# bash run_cifar10_v2.sh output 1 128 10 3000 128 s3 $1 &


# -------------------------------

bash run_cifar100_v7.sh output 128 128 100 3000 128 sample s3 $1 &
# bash run_cifar100_v7.sh output 128 128 100 1000 128 sample s3 $1 &

bash run_cifar100_v7.sh output 64 128 100 3000 128 sample s3 $1 &
bash run_cifar100_v7.sh output 64 128 100 1000 128 sample s3 $1 &

# bash run_cifar100_v7.sh output 16 128 100 3000 128 sample s3 $1 &
# bash run_cifar100_v7.sh output 16 128 100 1000 128 sample s3 $1 &

### 
bash run_cifar10_v3.sh output 64 128 100 3000 128 sample s3 $1 &
bash run_cifar10_v3.sh output 64 128 100 1000 128 sample s3 $1 &

# bash run_cifar10_v3.sh output 16 128 100 3000 128 sample s3 $1 &
# bash run_cifar10_v3.sh output 16 128 100 1000 128 sample s3 $1 &
