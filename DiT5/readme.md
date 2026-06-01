conda create -n dit_ttt python=3.12 -y
conda activate dit_ttt

pip install -r requirements.txt --timeout 1000

srun -J exp -N 1 -p RTX3090 -w node07  --gres gpu:8 torchrun --nproc_per_node 8 train.py  --model DiT-S/2  --data-path  /home/data/imagenet/train  --image-size 256 --global-batch-size 256
