torchrun --nproc_per_node=1 \       # For distribution across multiple GPUs

    ./src/tasks/train.py \

    --config ./src/configs/5frm_default.json \
    --per_gpu_train_batch_size 1 \
    --per_gpu_eval_batch_size 1 \
    --num_train_epochs 1 \
    --learning_rate 0.0001\
    --max_num_frames 5 \
    --backbone_coef_lr 0.05 \
    --learn_mask_enabled \
    --loss_sparse_w 0.5 \
    --lambda_ 0.1 \
    --output_dir ./output/output_exp1 \