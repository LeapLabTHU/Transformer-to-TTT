# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from losses import DistillationLoss
from samplers import RASampler
from augment import new_data_aug_generator

import models


import utils


def load_deit_weights_to_ttt(ttt_model, deit_checkpoint_path, strict=False, inherit_attn=True, inherit_mlp=True):
    """
    Load weights from a DeiT softmax model into a TTT model.
    
    Args:
        ttt_model: VisionTransformer instance with TTT attention.
        deit_checkpoint_path: Path to the DeiT checkpoint.pth file.
        strict: Whether to strictly match all weights.
        inherit_attn: Whether to inherit DeiT attention weights (qkv.weight, qkv.bias, proj.weight, proj.bias).
        inherit_mlp: Whether to inherit DeiT MLP weights (fc1, fc2).
    
    Returns:
        loaded_keys: List of successfully loaded weight keys.
        missing_keys: List of unloaded TTT-specific weight keys.
        unexpected_keys: List of DeiT-specific weight keys.
        newly_initialized_params: List of newly initialized parameter names, used for learning-rate grouping.
    """
    print(f"\n{'='*60}")
    print(f"Loading weights from DeiT model: {deit_checkpoint_path}")
    print(f"Inherit attention weights: {'yes' if inherit_attn else 'no'}")
    print(f"Inherit MLP weights: {'yes' if inherit_mlp else 'no'}")
    print(f"{'='*60}\n")
    
    # Load DeiT checkpoint
    checkpoint = torch.load(deit_checkpoint_path, map_location='cpu')
    
    # Extract state_dict
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            deit_state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            deit_state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            deit_state_dict = checkpoint['model_state_dict']
        else:
            deit_state_dict = checkpoint
    else:
        deit_state_dict = checkpoint
    
    # Remove the 'module.' prefix if the checkpoint was saved from DDP training
    cleaned_state_dict = {}
    for k, v in deit_state_dict.items():
        if k.startswith('module.'):
            cleaned_state_dict[k[7:]] = v
        else:
            cleaned_state_dict[k] = v
    deit_state_dict = cleaned_state_dict
    
    # Get the TTT model state_dict
    ttt_state_dict = ttt_model.state_dict()
    
    # Statistics
    loaded_keys = []
    missing_keys = []  # TTT-specific weights
    unexpected_keys = []  # DeiT-specific weights
    skipped_keys = []  # Weights skipped by user options
    shape_mismatch_keys = []
    newly_initialized_params = []  # List of newly initialized parameter names
    
    # Define TTT-specific weight patterns
    ttt_specific_patterns = [
        'attn.w1',
        'attn.w2',
        'attn.w3',
        'attn.scale',
    ]
    
    # Define DeiT-specific weight patterns for standard attention
    deit_specific_patterns = [
        'attn.qkv.weight',
        'attn.qkv.bias',
        'attn.proj.weight',
        'attn.proj.bias',
    ]
    
    # Define MLP weight patterns
    mlp_patterns = [
        'mlp.fc1.weight',
        'mlp.fc1.bias',
        'mlp.fc2.weight',
        'mlp.fc2.bias',
    ]
    
    def is_ttt_specific(key):
        """Return whether the key is TTT-specific."""
        return any(pattern in key for pattern in ttt_specific_patterns)
    
    def is_deit_specific(key):
        """Return whether the key is DeiT-specific."""
        return any(pattern in key for pattern in deit_specific_patterns)
    
    def is_mlp_weight(key):
        """Return whether the key is an MLP weight."""
        return any(pattern in key for pattern in mlp_patterns)
    
    def is_attn_weight(key):
        """Return whether the key is an attention weight, excluding norm parameters."""
        return 'attn.' in key and not 'norm' in key and not is_ttt_specific(key)
    
    # Create a new state_dict for loading
    new_state_dict = {}
    
    # Iterate over DeiT weights
    for deit_key, deit_value in deit_state_dict.items():
        # Check whether this weight should be skipped
        should_skip = False
        
        # Load attention weights according to inherit_attn
        if not inherit_attn and is_deit_specific(deit_key):
            skipped_keys.append(f"{deit_key} (attention weight - not inherited by user choice)")
            should_skip = True
        
        # Load MLP weights according to inherit_mlp
        if not inherit_mlp and is_mlp_weight(deit_key):
            skipped_keys.append(f"{deit_key} (MLP weight - not inherited by user choice)")
            should_skip = True
        
        if should_skip:
            continue
            
        if deit_key in ttt_state_dict:
            # Check shape compatibility
            if ttt_state_dict[deit_key].shape == deit_value.shape:
                new_state_dict[deit_key] = deit_value
                loaded_keys.append(deit_key)
            else:
                shape_mismatch_keys.append(
                    f"{deit_key}: DeiT{deit_value.shape} vs TTT{ttt_state_dict[deit_key].shape}"
                )
        else:
            if is_deit_specific(deit_key):
                unexpected_keys.append(deit_key)
    
    # Check weights not loaded in the TTT model
    for ttt_key in ttt_state_dict.keys():
        if ttt_key not in loaded_keys:
            if is_ttt_specific(ttt_key):
                missing_keys.append(f"{ttt_key} (TTT-specific)")
                newly_initialized_params.append(ttt_key)  # Add to the newly initialized list
            elif not inherit_attn and is_attn_weight(ttt_key):
                missing_keys.append(f"{ttt_key} (attention not inherited)")
                newly_initialized_params.append(ttt_key)  # Add to the newly initialized list
            elif not inherit_mlp and is_mlp_weight(ttt_key):
                missing_keys.append(f"{ttt_key} (MLP not inherited)")
                newly_initialized_params.append(ttt_key)  # Add to the newly initialized list
            elif ttt_key not in deit_state_dict:
                missing_keys.append(ttt_key)
                newly_initialized_params.append(ttt_key)  # Add to the newly initialized list
    
    # Load weights
    incompatible_keys = ttt_model.load_state_dict(new_state_dict, strict=False)
    
    # Print detailed information
    print(f"\n{'='*60}")
    print(f"Weight loading statistics")
    print(f"{'='*60}")
    print(f"✓ Successfully loaded: {len(loaded_keys)} weights")
    
    # Count loaded weight types
    loaded_attn = sum(1 for k in loaded_keys if is_attn_weight(k) or is_deit_specific(k))
    loaded_mlp = sum(1 for k in loaded_keys if is_mlp_weight(k))
    loaded_other = len(loaded_keys) - loaded_attn - loaded_mlp
    
    print(f"  - Base components (patch_embed, pos_embed, cls_token, norm, head, etc.): {loaded_other} weights")
    print(f"  - Attention weights: {loaded_attn} weights")
    print(f"  - MLP weights: {loaded_mlp} weights")
    
    if skipped_keys:
        print(f"\n⊘ Skipped by user choice: {len(skipped_keys)} entries")
        for key in skipped_keys[:10]:
            print(f"    - {key}")
        if len(skipped_keys) > 10:
            print(f"    ... and {len(skipped_keys)-10} more")
    
    print(f"\n⊗ DeiT-specific weights (skipped): {len(unexpected_keys)} entries")
    if unexpected_keys and len(unexpected_keys) <= 20:
        for key in unexpected_keys[:10]:
            print(f"    - {key}")
        if len(unexpected_keys) > 10:
            print(f"    ... and {len(unexpected_keys)-10} more")
    
    print(f"\n◆ Unloaded weights (kept randomly initialized): {len(missing_keys)} entries")
    if missing_keys:
        ttt_only = [k for k in missing_keys if is_ttt_specific(str(k))]
        if ttt_only and len(ttt_only) <= 20:
            print("  TTT attention-specific weights:")
            for key in ttt_only[:10]:
                print(f"    - {key}")
            if len(ttt_only) > 10:
                print(f"    ... and {len(ttt_only)-10} more")
        
        not_inherited = [k for k in missing_keys if not is_ttt_specific(str(k))]
        if not_inherited:
            print(f"\n  Weights not inherited: {len(not_inherited)} entries")
            for key in not_inherited[:10]:
                print(f"    - {key}")
            if len(not_inherited) > 10:
                print(f"    ... and {len(not_inherited)-10} more")
    
    if shape_mismatch_keys:
        print(f"\n⚠ Shape mismatches: {len(shape_mismatch_keys)} entries")
        for msg in shape_mismatch_keys[:5]:
            print(f"    - {msg}")
        if len(shape_mismatch_keys) > 5:
            print(f"    ... and {len(shape_mismatch_keys)-5} more")
    
    print(f"\n{'='*60}\n")
    
    # Count the loaded parameter ratio
    total_params = sum(p.numel() for p in ttt_model.parameters())
    loaded_params = sum(v.numel() for k, v in new_state_dict.items())
    print(f"Loaded parameter ratio: {loaded_params/total_params*100:.2f}% ({loaded_params:,}/{total_params:,})")
    
    # Print newly initialized parameter information
    print(f"\n{'='*60}")
    print(f"★ Newly initialized parameters (will use a higher learning rate): {len(newly_initialized_params)} parameters")
    print(f"{'='*60}")
    for param_name in newly_initialized_params[:20]:
        print(f"    - {param_name}")
    if len(newly_initialized_params) > 20:
        print(f"    ... and {len(newly_initialized_params)-20} more")
    
    return loaded_keys, missing_keys, unexpected_keys, newly_initialized_params


def create_optimizer_with_layer_lr(args, model, newly_initialized_params, lr_multiplier=10.0):
    """
    Create an optimizer that uses a higher learning rate for newly initialized parameters.
    
    Args:
        args: Argument configuration.
        model: Model.
        newly_initialized_params: List of newly initialized parameter names.
        lr_multiplier: Learning-rate multiplier for newly initialized parameters.
    
    Returns:
        optimizer: Optimizer.
    """
    # Convert newly initialized parameter names to a set for lookup
    newly_init_set = set(newly_initialized_params)
    
    # Split parameter groups
    pretrained_params = []
    new_params = []
    
    pretrained_params_names = []
    new_params_names = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Check whether this parameter is newly initialized
        # Handle possible suffixes such as "(TTT-specific)"
        is_new = False
        for new_param_name in newly_init_set:
            # Remove possible marker suffixes
            clean_name = new_param_name.split(' ')[0] if ' ' in new_param_name else new_param_name
            if name == clean_name:
                is_new = True
                break
        
        if is_new:
            new_params.append(param)
            new_params_names.append(name)
        else:
            pretrained_params.append(param)
            pretrained_params_names.append(name)
    
    # Print parameter group information
    print(f"\n{'='*60}")
    print(f"Optimizer parameter groups")
    print(f"{'='*60}")
    print(f"Pretrained parameters (base_lr={args.lr}): {len(pretrained_params)} tensors")
    print(f"Newly initialized parameters (lr={args.lr * lr_multiplier}): {len(new_params)} tensors")
    
    if new_params_names:
        print(f"\nNewly initialized parameter list (using {lr_multiplier}x learning rate):")
        for name in new_params_names[:20]:
            print(f"    - {name}")
        if len(new_params_names) > 20:
            print(f"    ... and {len(new_params_names)-20} more")
    print(f"{'='*60}\n")
    
    # Create parameter groups
    param_groups = [
        {
            'params': pretrained_params,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'param_names': pretrained_params_names,  # for debugging
        },
        {
            'params': new_params,
            'lr': args.lr * lr_multiplier,
            'weight_decay': args.weight_decay,
            'param_names': new_params_names,  # for debugging
        }
    ]
    
    # Create the optimizer according to the optimizer type
    opt_lower = args.opt.lower()
    
    if opt_lower == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            momentum=args.momentum,
            nesterov=True
        )
    elif opt_lower == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            eps=args.opt_eps,
            betas=args.opt_betas if args.opt_betas else (0.9, 0.999)
        )
    elif opt_lower == 'adamw':
        optimizer = torch.optim.AdamW(
            param_groups,
            eps=args.opt_eps,
            betas=args.opt_betas if args.opt_betas else (0.9, 0.999)
        )
    else:
        # Fall back to timm's create_optimizer
        # This will lose the grouped learning-rate behavior
        print(f"Warning: optimizer {args.opt} does not support grouped learning rates; using a single learning rate")
        optimizer = create_optimizer(args, model)
    
    return optimizer


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--unscale-lr', action='store_true')

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=2e-5, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=2e-8, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=2e-7, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Learning-rate multiplier for newly initialized parameters
    parser.add_argument('--new-params-lr-multiplier', type=float, default=20.0,
                        help='Learning rate multiplier for newly initialized parameters (default: 10.0)')
    # Whether to freeze pretrained weights and train only newly initialized parameters
    parser.add_argument('--freeze-pretrained', action='store_true', default=False,
                        help='Freeze pretrained weights and only train newly initialized parameters')
    # Other parameters remain unchanged...
    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)

    parser.add_argument('--ThreeAugment', action='store_true') #3augment

    parser.add_argument('--src', action='store_true') #simple random crop

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='deit_tiny_patch16_224', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='teacher_checkpoint.pth')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # * Cosub params
    parser.add_argument('--cosub', action='store_true') 

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--attn-only', action='store_true')
    
    # Load weights from a DeiT model into the TTT model
    parser.add_argument('--load-deit-to-ttt', default='deit_tiny.pth', type=str,
                        help='Load DeiT softmax model weights to TTT model (specify DeiT checkpoint path)')
    parser.add_argument('--inherit_attn', action='store_true', default=True,
                        help='Whether to inherit attention-layer weights from DeiT (qkv, proj)')
    parser.add_argument('--inherit_mlp', action='store_true', default=True,
                        help='Whether to inherit MLP-layer weights from DeiT (fc1, fc2)')

    # Dataset parameters
    parser.add_argument('--data-path', default='data/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=False)

    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)

    print(args)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    if args.ThreeAugment:
        data_loader_train.dataset.transform = new_data_aug_generator(args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=args.input_size
    )

    # Store newly initialized parameter names
    newly_initialized_params = []

    # Load weights from the DeiT model into the TTT model
    if args.load_deit_to_ttt:
        print("\n" + "="*60)
        print("Use --load-deit-to-ttt to initialize the TTT model from a DeiT checkpoint")
        print("="*60)
        loaded_keys, missing_keys, unexpected_keys, newly_initialized_params = load_deit_weights_to_ttt(
            model, args.load_deit_to_ttt, strict=False,
            inherit_attn=args.inherit_attn, inherit_mlp=args.inherit_mlp
        )
        

        # Freeze pretrained weights if requested
        if args.freeze_pretrained and newly_initialized_params:
            print("\n" + "="*60)
            print("Freeze pretrained weights and train only newly initialized parameters")
            print("="*60)
            
            # Convert newly initialized parameter names to a set
            newly_init_set = set()
            for param_name in newly_initialized_params:
                clean_name = param_name.split(' ')[0] if ' ' in param_name else param_name
                newly_init_set.add(clean_name)
            
            frozen_count = 0
            trainable_count = 0
            for name, param in model.named_parameters():
                if name in newly_init_set:
                    param.requires_grad = True
                    trainable_count += 1
                else:
                    param.requires_grad = False
                    frozen_count += 1
            
            print(f"Number of frozen parameters: {frozen_count}")
            print(f"Number of trainable parameters: {trainable_count}")
            print("="*60 + "\n")

        # Evaluate the model immediately after loading weights
        print("\n" + "="*60)
        print("Evaluate the TTT model after loading DeiT weights...")
        print("="*60)

        # Move the model to the device for evaluation
        model.to(device)
        model.eval()

        # Run evaluation
        with torch.no_grad():
            test_stats = evaluate(data_loader_val, model, device)

        print(f"\nAccuracy after loading DeiT weights: {test_stats['acc1']:.2f}%")
        print(f"Top-5 accuracy: {test_stats['acc5']:.2f}%")
        print(f"Loss: {test_stats['loss']:.4f}")
        print("="*60 + "\n")

        # Move the model back to CPU for the normal training flow
        model.cpu()
    
    # Keep the original finetuning logic unchanged
    elif args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)
        
    if args.attn_only:
        for name_p,p in model.named_parameters():
            if '.attn.' in name_p:
                p.requires_grad = True
            else:
                p.requires_grad = False
        try:
            model.head.weight.requires_grad = True
            model.head.bias.requires_grad = True
        except:
            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True
        try:
            model.pos_embed.requires_grad = True
        except:
            print('no position encoding')
        try:
            for p in model.patch_embed.parameters():
                p.requires_grad = False
        except:
            print('no patch embed')
            
    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
        args.lr = linear_scaled_lr
    
    # Choose optimizer construction based on whether newly initialized parameters exist
    if args.load_deit_to_ttt and newly_initialized_params:
        print(f"\nUsing grouped learning-rate optimizer: new parameters use {args.new_params_lr_multiplier}x learning rate")
        optimizer = create_optimizer_with_layer_lr(
            args, model_without_ddp, newly_initialized_params, 
            lr_multiplier=args.new_params_lr_multiplier
        )
    else:
        optimizer = create_optimizer(args, model_without_ddp)
    
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        
    if args.bce_loss:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            # global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
        lr_scheduler.step(args.start_epoch)
        
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    print(f"Start training for {args.epochs} epochs")
    
    start_time = time.time()
    max_accuracy = 0.0
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        lr_scheduler.step(epoch)
        
        lr_info = f"Epoch {epoch}/{args.epochs-1}, LR: {optimizer.param_groups[0]['lr']:.2e}"
        if len(optimizer.param_groups) > 1:
            lr_info += f" (new params: {optimizer.param_groups[1]['lr']:.2e})"
        print(lr_info)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.train_mode,
            args = args,
        )

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                    'newly_initialized_params': newly_initialized_params,
                }, checkpoint_path)

        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        
        if max_accuracy < test_stats["acc1"]:
            max_accuracy = test_stats["acc1"]
            if args.output_dir:
                checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                        'newly_initialized_params': newly_initialized_params,
                    }, checkpoint_path)
            
        print(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters,
                     'lr': optimizer.param_groups[0]['lr'],
                     'lr_new_params': optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else optimizer.param_groups[0]['lr']}
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    print(f"\n{'='*60}")
    print(f"Training finished. Total epochs: {args.epochs}")
    if len(optimizer.param_groups) > 1:
        print(f"Newly initialized parameters use {args.new_params_lr_multiplier}x learning rate")
    print(f"Best accuracy: {max_accuracy:.2f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
