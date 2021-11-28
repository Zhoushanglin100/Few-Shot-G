### moment loss

# export ld=1000
# export eN=2000
# export eG=50
# export lr_G=0.001
export lr_S=0.06
export dcy=5

for lr_G in 0.01 0.001 0.0001
do
    for eG in 50 100 200
    do
        let eN=eG*40

        for ld in 1000 1500 2000 2500 3000
        do
            python3 DAFLDeepinvert-train_v4.py \
                            --dataset cifar10 --channels 3 \
                            --n_epochs $eN --n_epochs_G $eG \
                            --fix_G --batch_size 256 \
                            --lr_G $lr_G --lr_S $lr_S --decay $dcy \
                            --latent_dim $ld \
                            --resume &
        done
    done
done


### fix G

for lr_G in 0.01 0.001 0.0001
do
    for eG in 50 100 200
    do
        let eN=eG*40

        for ld in 1000 1500 2000 2500 3000
        do
            python3 DAFLDeepinvert-train_v4.py \
                            --dataset cifar10 --channels 3 \
                            --n_epochs $eN --n_epochs_G $eG \
                            --batch_size 256 \
                            --lr_G $lr_G --lr_S $lr_S --decay $dcy \
                            --latent_dim $ld \
                            --resume &
        done
    done
done
