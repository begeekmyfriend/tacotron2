# CUDA_VISIBLE_DEVICES=3 python train.py -o logs --init-lr 1e-3 --final-lr 1e-5 --epochs 1000 -bs 32 --weight-decay 1e-6 --grad-clip-thresh 1.0 --cudnn-enabled --log-file nvlog.json
CUDA_VISIBLE_DEVICES=3 nohup python train.py -o logs --init-lr 1e-3 --final-lr 1e-5 --epochs 1000 -bs 32 --weight-decay 1e-6 --grad-clip-thresh 1.0 --cudnn-enabled --log-file nvlog.json > leoma.out &
