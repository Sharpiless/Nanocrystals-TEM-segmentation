python train_ours.py \
    --data-dir=data \
    --device=cuda \
    --batch_size=8 \
    --epoch=100 \
    --save_path=runs/extra0312 \
    --labeled_data \
    --ckpt_path=../ckpt/epoch28.pth