# 单卡
# P5 models
python tools/train.py --batch 32 --config configs/yolov6s_finetune.py --data data/dataset.yaml --fuse_ab --device 0
# P6 models
python tools/train.py --batch 32 --config configs/yolov6s6_finetune.py --data data/dataset.yaml --img 1280 --device 0

# 多卡
# P5 models
python -m torch.distributed.launch --nproc_per_node 8 tools/train.py --batch 256 --conf configs/yolov6s_finetune.py -- data data/dataset.yaml --fuse_ab --device 0,1,2,3,4,5,6,7
# P6 models
python -m torch.distributed.launch --nproc_per_node 8 tools/train.py --batch 128 --conf configs/yolov6s6_finetune.py --data data/dataset.yaml --img 1280 --device 0,1,2,3,4,5,6,7

# ------------------------------------------------------------
# 如果您的训练进程中断了，您可以这样恢复先前的训练进程
# 命令将自动在 YOLOv6 当前训练保存目录中找到最近保存的模型，然后恢复训练
# ------------------------------------------------------------
# 单卡训练
python tools/train.py --resume /path/to/your/checkpoint/path
# 多卡训练
python -m torch.distributed.launch --nproc_per_node 8 tools/train.py --resume /path/to/your/checkpoint/path


# tensorboard
tensorboard --logdir=your_path/to/log
