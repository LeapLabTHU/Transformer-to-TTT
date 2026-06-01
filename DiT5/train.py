# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def create_optimizer_with_layer_lr(model, base_lr=1e-4, ttt_lr_scale=10.0, pretrained_lr_scale=0.1, logger=None):
    """
    Set different learning rates for different parameter types.
    
    Args:
        model: Model instance.
        base_lr: Base learning rate.
        ttt_lr_scale: Learning-rate multiplier for TTT-specific parameters, relative to base_lr.
        pretrained_lr_scale: Learning-rate multiplier for pretrained parameters, relative to base_lr.
        logger: Logger instance.
    
    Returns:
        optimizer: Optimizer configured with layer-wise learning rates.
    """
    def log_info(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    # Patterns for TTT-specific parameters
    ttt_specific_patterns = ['attn.w1', 'attn.w2', 'attn.b1', 'attn.b2','attn.w3']
    
    # Group parameters
    ttt_params = []           # TTT-specific parameters (randomly initialized)
    ttt_param_names = []
    pretrained_params = []    # Inherited pretrained parameters
    pretrained_param_names = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        is_ttt_specific = any(pattern in name for pattern in ttt_specific_patterns)
        
        if is_ttt_specific:
            ttt_params.append(param)
            ttt_param_names.append(name)
        else:
            pretrained_params.append(param)
            pretrained_param_names.append(name)
    
    # Create parameter groups
    param_groups = []
    
    if ttt_params:
        param_groups.append({
            'params': ttt_params,
            'lr': base_lr * ttt_lr_scale,
            'name': 'ttt_specific'
        })
    
    if pretrained_params:
        param_groups.append({
            'params': pretrained_params,
            'lr': base_lr * pretrained_lr_scale,
            'name': 'pretrained'
        })
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0)
    
    # Print detailed information
    log_info(f"\n{'='*60}")
    log_info(f"Layer-wise learning-rate optimizer configuration")
    log_info(f"{'='*60}")
    log_info(f"Base learning rate: {base_lr}")
    log_info(f"Learning rate for TTT-specific parameters: {base_lr * ttt_lr_scale:.2e} (scale: {ttt_lr_scale}x)")
    log_info(f"Learning rate for pretrained parameters: {base_lr * pretrained_lr_scale:.2e} (scale: {pretrained_lr_scale}x)")
    log_info(f"")
    log_info(f"Number of TTT-specific parameters: {len(ttt_params)}")
    log_info(f"  Parameter count: {sum(p.numel() for p in ttt_params):,}")
    
    # Show a subset of TTT parameter names
    if ttt_param_names:
        log_info(f"  Example parameters:")
        for name in ttt_param_names[:5]:
            log_info(f"    - {name}")
        if len(ttt_param_names) > 5:
            log_info(f"    ... {len(ttt_param_names)-5} more")
    
    log_info(f"")
    log_info(f"Number of pretrained parameters: {len(pretrained_params)}")
    log_info(f"  Parameter count: {sum(p.numel() for p in pretrained_params):,}")
    log_info(f"{'='*60}\n")
    
    return optimizer


def load_softmax_weights_to_ttt(ttt_model, softmax_checkpoint_path, strict=False, inherit_attn=True, inherit_mlp=True, logger=None):
    """
    Load weights from a standard Softmax Attention model (DiT/ViT) into a DiT TTT model.
    
    Args:
        ttt_model: DiT model instance with TTT attention.
        softmax_checkpoint_path: Checkpoint path of the standard Softmax model.
        strict: Whether to strictly match all weights.
        inherit_attn: Whether to inherit standard Attention-layer weights (qkv.weight, qkv.bias, proj.weight, proj.bias).
        inherit_mlp: Whether to inherit MLP-layer weights (fc1, fc2, or weights inside mlp).
        logger: Logger instance.
    
    Returns:
        loaded_keys: List of successfully loaded weight keys.
        missing_keys: List of TTT-specific or otherwise unloaded weight keys.
        unexpected_keys: List of Softmax-model-specific weight keys.
    """
    def log_info(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    log_info(f"\n{'='*60}")
    log_info(f"Loading weights from the standard Softmax Attention model: {softmax_checkpoint_path}")
    log_info(f"Inherit Attention weights: {'yes' if inherit_attn else 'no'}")
    log_info(f"Inherit MLP weights: {'yes' if inherit_mlp else 'no'}")
    log_info(f"{'='*60}\n")
    
    # Load checkpoint
    checkpoint = torch.load(softmax_checkpoint_path, map_location='cpu')
    
    # Extract state_dict
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            softmax_state_dict = checkpoint['model']
            print("Using model weights from the checkpoint.")
        elif 'state_dict' in checkpoint:
            softmax_state_dict = checkpoint['state_dict']
            print("Using 'state_dict' from the checkpoint.")
        elif 'ema' in checkpoint:
            softmax_state_dict = checkpoint['ema']
            print("Using 'ema' weights from the checkpoint.")
        else:
            softmax_state_dict = checkpoint
    else:
        softmax_state_dict = checkpoint
    
    # Remove the 'module.' prefix if the checkpoint was trained with DDP
    cleaned_state_dict = {}
    for k, v in softmax_state_dict.items():
        if k.startswith('module.'):
            cleaned_state_dict[k[7:]] = v
        else:
            cleaned_state_dict[k] = v
    softmax_state_dict = cleaned_state_dict
    
    # Get the TTT model state_dict
    ttt_state_dict = ttt_model.state_dict()
    
    # Statistics
    loaded_keys = []
    missing_keys = []  # TTT-specific weights or weights that were not inherited
    unexpected_keys = []  # Weights specific to the Softmax model
    skipped_keys = []  # Weights skipped because of argument settings
    shape_mismatch_keys = []
    
    # Define TTT-specific weight patterns
    ttt_specific_patterns = [
        'attn.w1',
        'attn.w2',
    ]
    
    # Define standard Attention weight patterns (qkv, proj, etc.)
    standard_attn_patterns = [
        'attn.qkv.weight',
        'attn.qkv.bias',
        'attn.proj.weight',
        'attn.proj.bias',
    ]
    
    # Define MLP weight patterns (DiT uses timm Mlp with fc1 and fc2)
    mlp_patterns = [
        'mlp.fc1.weight',
        'mlp.fc1.bias',
        'mlp.fc2.weight',
        'mlp.fc2.bias',
    ]
    
    # Define base component patterns (patch_embed, pos_embed, norm, etc.)
    base_component_patterns = [
        'x_embedder',      # DiT-specific
        't_embedder',      # DiT-specific
        'y_embedder',      # DiT-specific
        'pos_embed',
        'cls_token',
        'dist_token',
        'patch_embed',
        'final_layer',     # DiT-specific
        'adaLN_modulation', # DiT-specific
        'norm',
        'head',
    ]
    
    def is_ttt_specific(key):
        """Return whether the key is a TTT-specific weight."""
        return any(pattern in key for pattern in ttt_specific_patterns)
    
    def is_standard_attn(key):
        """Return whether the key is a standard attention weight."""
        return any(pattern in key for pattern in standard_attn_patterns)
    
    def is_mlp_weight(key):
        """Return whether the key is an MLP weight."""
        return any(pattern in key for pattern in mlp_patterns)
    
    def is_base_component(key):
        """Return whether the key is a base component weight."""
        return any(pattern in key for pattern in base_component_patterns)
    
    def is_norm_weight(key):
        """Return whether the key is a normalization-layer weight."""
        return 'norm' in key and 'attn' not in key and 'mlp' not in key
    
    # Create a new state_dict for loading
    new_state_dict = {}
    
    # Iterate over the weights of the Softmax model
    for softmax_key, softmax_value in softmax_state_dict.items():
        # Check whether this weight should be skipped
        should_skip = False
        skip_reason = ""
        
        # Skip TTT-specific weights, which should not appear in a Softmax model
        if is_ttt_specific(softmax_key):
            unexpected_keys.append(f"{softmax_key} (This should not appear in the Softmax model)")
            continue
        
        # Decide whether to load attention weights according to inherit_attn
        if not inherit_attn and is_standard_attn(softmax_key):
            skipped_keys.append(f"{softmax_key} (standard Attention weights - user chose not to inherit them)")
            should_skip = True
            skip_reason = "User chose not to inherit Attention weights"
        
        # Decide whether to load MLP weights according to inherit_mlp
        if not inherit_mlp and is_mlp_weight(softmax_key):
            skipped_keys.append(f"{softmax_key} (MLP weights - user chose not to inherit them)")
            should_skip = True
            skip_reason = "User chose not to inherit MLP weights"
        
        if should_skip:
            continue
            
        # Check whether this weight exists in the TTT model
        if softmax_key in ttt_state_dict:
            # Check whether the tensor shapes match
            if ttt_state_dict[softmax_key].shape == softmax_value.shape:
                new_state_dict[softmax_key] = softmax_value
                loaded_keys.append(softmax_key)
            else:
                shape_mismatch_keys.append(
                    f"{softmax_key}: Softmax{softmax_value.shape} vs TTT{ttt_state_dict[softmax_key].shape}"
                )
        else:
            # This weight does not exist in the TTT model
            unexpected_keys.append(f"{softmax_key} (not present in the TTT model)")
    
    # Check weights in the TTT model that were not loaded
    for ttt_key in ttt_state_dict.keys():
        if ttt_key not in loaded_keys:
            if is_ttt_specific(ttt_key):
                missing_keys.append(f"{ttt_key} (TTT-specific weights)")
            elif not inherit_attn and is_standard_attn(ttt_key):
                missing_keys.append(f"{ttt_key} (User chose not to inherit Attention weights)")
            elif not inherit_mlp and is_mlp_weight(ttt_key):
                missing_keys.append(f"{ttt_key} (User chose not to inherit MLP weights)")
            elif ttt_key not in softmax_state_dict:
                missing_keys.append(f"{ttt_key} (not present in the Softmax model)")
    
    # Load weights
    incompatible_keys = ttt_model.load_state_dict(new_state_dict, strict=False)
    
    # Print detailed information
    log_info(f"\n{'='*60}")
    log_info(f"Weight loading statistics")
    log_info(f"{'='*60}")
    log_info(f"✓ Successfully loaded: {len(loaded_keys)} weights")
    
    # Count loaded weights by type
    loaded_attn = sum(1 for k in loaded_keys if is_standard_attn(k))
    loaded_mlp = sum(1 for k in loaded_keys if is_mlp_weight(k))
    loaded_base = sum(1 for k in loaded_keys if is_base_component(k))
    loaded_norm = sum(1 for k in loaded_keys if is_norm_weight(k))
    loaded_other = len(loaded_keys) - loaded_attn - loaded_mlp - loaded_base - loaded_norm
    
    log_info(f"  - Base components (x_embedder, t_embedder, y_embedder, pos_embed, final_layer, etc.): {loaded_base}")
    log_info(f"  - Normalization layers: {loaded_norm}")
    log_info(f"  - Standard Attention weights (qkv, proj): {loaded_attn}")
    log_info(f"  - MLP weights (fc1, fc2): {loaded_mlp}")
    if loaded_other > 0:
        log_info(f"  - Other weights: {loaded_other}")
    
    if skipped_keys:
        log_info(f"\n⊘ User chose not to inherit: {len(skipped_keys)}")
        for key in skipped_keys[:10]:
            log_info(f"    - {key}")
        if len(skipped_keys) > 10:
            log_info(f"    ... {len(skipped_keys)-10} more")
    
    log_info(f"\n⊗ Weights specific to the Softmax model (not present in TTT): {len(unexpected_keys)}")
    if unexpected_keys and len(unexpected_keys) <= 20:
        for key in unexpected_keys[:10]:
            log_info(f"    - {key}")
        if len(unexpected_keys) > 10:
            log_info(f"    ... {len(unexpected_keys)-10} more")
    
    log_info(f"\n◆ Unloaded weights (kept randomly initialized): {len(missing_keys)}")
    if missing_keys:
        ttt_only = [k for k in missing_keys if 'TTT-specific weights' in str(k)]
        if ttt_only:
            log_info(f"  TTT Attention-specific weights (w1, w2): {len(ttt_only)}")
            for key in ttt_only[:10]:
                log_info(f"    - {key}")
            if len(ttt_only) > 10:
                log_info(f"    ... {len(ttt_only)-10} more")
        
        not_inherited = [k for k in missing_keys if 'TTT-specific weights' not in str(k)]
        if not_inherited:
            log_info(f"\n  Other non-inherited weights: {len(not_inherited)}")
            for key in not_inherited[:10]:
                log_info(f"    - {key}")
            if len(not_inherited) > 10:
                log_info(f"    ... {len(not_inherited)-10} more")
    
    if shape_mismatch_keys:
        log_info(f"\n⚠ Shape mismatches (skipped): {len(shape_mismatch_keys)}")
        for msg in shape_mismatch_keys[:5]:
            log_info(f"    - {msg}")
        if len(shape_mismatch_keys) > 5:
            log_info(f"    ... {len(shape_mismatch_keys)-5} more")
    
    log_info(f"\n{'='*60}\n")
    
    # Compute the proportion of loaded parameters
    total_params = sum(p.numel() for p in ttt_model.parameters())
    loaded_params = sum(v.numel() for k, v in new_state_dict.items())
    log_info(f"Parameter loading ratio: {loaded_params/total_params*100:.2f}% ({loaded_params:,}/{total_params:,})")
    log_info(f"Note: TTT-specific w1 and w2 parameters will remain randomly initialized\n")
    
    return loaded_keys, missing_keys, unexpected_keys


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def interpolate_state_dict(ckpt_state_dict, current_state_dict):
    """
    size mismatch for pos_embed: copying a param with shape torch.Size([1, 256, 384]) from checkpoint, the shape in current model is torch.Size([1, 1024, 384]).
    size mismatch for blocks.0.attn.ah_bias: copying a param with shape torch.Size([1, 6, 4, 16, 1]) from checkpoint, the shape in current model is torch.Size([1, 6, 4, 32, 1]).
    size mismatch for blocks.0.attn.aw_bias: copying a param with shape torch.Size([1, 6, 4, 1, 16]) from checkpoint, the shape in current model is torch.Size([1, 6, 4, 1, 32]).
    size mismatch for blocks.0.attn.ha_bias: copying a param with shape torch.Size([1, 6, 16, 1, 4]) from checkpoint, the shape in current model is torch.Size([1, 6, 32, 1, 4]).
    size mismatch for blocks.0.attn.wa_bias: copying a param with shape torch.Size([1, 6, 1, 16, 4]) from checkpoint, the shape in current model is torch.Size([1, 6, 1, 32, 4]).
    """
    state_dict_after_interp = dict()

    for key, value in ckpt_state_dict.items():
        if ("pos_embed" == key):
            # fixed sin-cos embedding
            state_dict_after_interp[key] = current_state_dict[key]
        elif ("attn.ah_bias" in key):
            input_tensor = ckpt_state_dict[key].squeeze(0).squeeze(-1)  # [1, 6, 4, 16, 1] --> [6, 4, 16]
            output_spatial_size = current_state_dict[key].shape[-2]  # [1, 6, 4, 32, 1] --> 32
            interpolated = F.interpolate(input_tensor, size=output_spatial_size, mode='linear', align_corners=True)  #  [6, 4, 16] --> [6, 4, 32]
            state_dict_after_interp[key] = interpolated.unsqueeze(0).unsqueeze(-1)  # [6, 4, 32] --> [1, 6, 4, 32, 1]
        elif ("attn.aw_bias" in key):
            input_tensor = ckpt_state_dict[key].squeeze(0).squeeze(-2)  # [1, 6, 4, 1, 16] --> [6, 4, 16]
            output_spatial_size = current_state_dict[key].shape[-1]  # [1, 6, 4, 1, 32] --> 32
            interpolated = F.interpolate(input_tensor, size=output_spatial_size, mode='linear', align_corners=True)  #  [6, 4, 16] --> [6, 4, 32]
            state_dict_after_interp[key] = interpolated.unsqueeze(0).unsqueeze(-2)  # [6, 4, 32] --> [1, 6, 4, 1, 32]
        elif ("attn.ha_bias" in key):
            input_tensor = ckpt_state_dict[key].squeeze(0).squeeze(-2).permute(0, 2, 1)  # [1, 6, 16, 1, 4] --> [6, 16, 4] --> [6, 4, 16]
            output_spatial_size = current_state_dict[key].shape[-3]  # [1, 6, 32, 1, 4] --> 32
            interpolated = F.interpolate(input_tensor, size=output_spatial_size, mode='linear', align_corners=True)  #  [6, 4, 16] --> [6, 4, 32]
            state_dict_after_interp[key] = interpolated.permute(0, 2, 1).unsqueeze(0).unsqueeze(-2)  # [6, 4, 32] --> [6, 32, 4] --> [1, 6, 32, 1, 4]
        elif ("attn.wa_bias" in key):
            input_tensor = ckpt_state_dict[key].squeeze(0).squeeze(-3).permute(0, 2, 1)  # [1, 6, 1, 16, 4] --> [6, 16, 4] --> [6, 4, 16]
            output_spatial_size = current_state_dict[key].shape[-2]  # [1, 6, 1, 32, 4] --> 32
            interpolated = F.interpolate(input_tensor, size=output_spatial_size, mode='linear', align_corners=True)  #  [6, 4, 16] --> [6, 4, 32]
            state_dict_after_interp[key] = interpolated.permute(0, 2, 1).unsqueeze(0).unsqueeze(-3)  # [6, 4, 32] --> [6, 32, 4] --> [1, 6, 4, 32, 1]
        else:
            state_dict_after_interp[key] = ckpt_state_dict[key]
    
    return state_dict_after_interp


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    
    # Add this block after creating the model and before wrapping it with DDP:
    if args.gradient_checkpointing:
        logger.info("Enabled Gradient Checkpointing mode to significantly reduce memory usage")
        model.gradient_checkpointing = True   # The DiT model natively supports this.
    # *** Load weights from a standard Softmax Attention model into the TTT model ***
    if args.load_softmax_to_ttt:
        logger.info("\n" + "="*60)
        logger.info("Using --load-softmax-to-ttt to initialize the TTT model from a standard Softmax Attention model")
        logger.info("="*60)
        load_softmax_weights_to_ttt(
            model, 
            args.load_softmax_to_ttt, 
            strict=False,
            inherit_attn=args.inherit_attn,
            inherit_mlp=args.inherit_mlp,
            logger=logger
        )
    
    # Load pretrained weights if specified:
    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        logger.info(f"Loaded pretrained weights from {args.ckpt}")
        
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    
    # Load EMA weights if checkpoint was loaded
    if args.ckpt:
        ema.load_state_dict(checkpoint["ema"], strict=False)
    
    


    model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ============ Setup optimizer ============
    # Select the optimizer configuration according to whether layer-wise learning rates are enabled
    if args.load_softmax_to_ttt and args.use_layer_lr:
        # Fine-tuning mode: use layer-wise learning rates
        logger.info("\n" + "="*60)
        logger.info("Enabled layer-wise learning-rate optimizer (--use-layer-lr)")
        logger.info("="*60)
        opt = create_optimizer_with_layer_lr(
            model.module,  # Note: use .module for a DDP-wrapped model
            base_lr=args.lr,
            ttt_lr_scale=args.ttt_lr_scale,
            pretrained_lr_scale=args.pretrained_lr_scale,
            logger=logger
        )
    else:
        # Regular mode: use a unified learning rate
        # opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
        # logger.info(f"Using a unified learning rate: {args.lr}")
        # Optimizer
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=2e-4,              # Initial maximum learning rate
            weight_decay=0.0
        )

        # Cosine Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=args.epochs,   # Or total epochs
            eta_min=1e-4              # Minimum learning rate
        )

        logger.info("Using CosineAnnealingLR: decaying lr from 2e-4 to 1e-4")

    
    # Load optimizer state if checkpoint was loaded
    if args.ckpt:
        opt.load_state_dict(checkpoint["opt"])

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    # Resume a pretrained low-resolution model
    if (args.resume_ckpt_low_resolution is not None) and (os.path.exists(args.resume_ckpt_low_resolution)):
        logger.info(f'Start resume from {args.resume_ckpt_low_resolution}')
        ckpt = torch.load(args.resume_ckpt_low_resolution, map_location='cpu')
        args.start_epoch = 0

        # interpolate the pos_embed, ah_bias, aw_bias, ha_bias, wa_bias
        state_dict_after_interp_model = interpolate_state_dict(ckpt['model'], current_state_dict=model.module.state_dict())
        state_dict_after_interp_ema = interpolate_state_dict(ckpt['ema'], current_state_dict=ema.state_dict())

        model.module.load_state_dict(state_dict_after_interp_model)
        ema.load_state_dict(state_dict_after_interp_ema)
        
        logger.info(f'Finish resume from low resolution, from finished epoch {args.start_epoch}.')

    requires_grad(ema, False)
    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            # scheduler.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                
                # Get current learning-rate information
                if args.load_softmax_to_ttt and args.use_layer_lr:
                    lr_info = ", ".join([f"LR-{g.get('name', i)}: {g['lr']:.2e}" for i, g in enumerate(opt.param_groups)])
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, {lr_info}")
                else:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, LR: {args.lr:.2e}")
                
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="/results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512, 1024], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400,help="1400")
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=1000000)
    parser.add_argument("--ckpt", type=str, default=None, help="Path to pretrained model weights.")
    parser.add_argument("--resume_ckpt_low_resolution", type=str, help="ckpt to resume")

    # *** Load weights from a standard Softmax Attention model into the TTT model ***
    parser.add_argument('--load-softmax-to-ttt', default='Softmax-S2-256.pt', type=str,
                        help='Load weights from a standard Softmax Attention model, such as the original DiT, into the TTT model by specifying the checkpoint path')
    parser.add_argument('--inherit-attn', action='store_true', default=True,
                        help='Whether to inherit Attention-layer weights from the Softmax model (qkv, proj, etc.)')
    parser.add_argument('--inherit-mlp', action='store_true', default=True,
                        help='Whether to inherit MLP-layer weights from the Softmax model (fc1, fc2)')
    
    # *** New: layer-wise learning-rate arguments ***
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Base learning rate (default: 1e-4)')
    parser.add_argument('--use-layer-lr', action='store_true', default=True,
                        help='Whether to enable layer-wise learning rates (only effective with --load-softmax-to-ttt)')
    parser.add_argument('--ttt-lr-scale', type=float, default=2.0,
                        help='Learning-rate multiplier for TTT-specific parameters (w1, w2), relative to the base learning rate (default: 10.0)')
    parser.add_argument('--pretrained-lr-scale', type=float, default=1.0,
                        help='Learning-rate multiplier for pretrained parameters, relative to the base learning rate (default: 0.1)')
    
    parser.add_argument('--gradient-checkpointing', action='store_true', default=False,
                    help='Whether to enable gradient checkpointing to save memory')
    args = parser.parse_args()
    main(args)