
python /data3/lwp/work/pytorch-CycleGAN-and-pix2pix-master/simclr_custom/main_mist_her2_ihc.py \
    --task simclr_mist_her2_ihc_512 \
    --model_type resnet18 \
    --out_dim 512 \
    --num_train_sample 50000 \
    --num_val_sample 5000 \
    --pos_ratio 0.5 \
    --max_epoch 30 \
    --early_stop 10 \
    --lr 0.00001 \
    --seed 1 \
    --batch_size 16 \
    --num_worker 1 \
    --device 0

export PYTHONPATH="${PYTHONPATH}:/data3/lwp/work/pytorch-CycleGAN-and-pix2pix-master"
