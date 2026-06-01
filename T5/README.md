# Linearizing Vision Transformer with Test Time Training(ICML 2026)

deit_tiny_checkpoint(weights to inherit from):https://cloud.tsinghua.edu.cn/f/8efdb83ccae8422bb0ad/?dl=1

The enviroment is the same as DeiT

conda create -n t5 python=3.10 -y

conda activate t5

pip install -r requirements.txt

python -m torch.distributed.launch --nproc_per_node=4  --use_env  --master_port=1321  main_30+10_lr.py --model draft_deit_tiny  --batch-size 256 --data-path /home/data/imagenet --output_dir /cluster/nvme6b/lyn/Linearizing-Vision-Transformers-with-Test-Time-Training-main/results
