import argparse
import os
import shutil
import sys
import time
from functools import partial
from typing import List, Optional
import deepspeed
import numpy as np
import torch
import tqdm
import json
import transformers
import re
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

import hashlib

from pll import (
    PLLForCausalLM, 
    TrainDataset, 
    ValDataset, 
    collate_fn, 
    evaluate_inst, 
    evaluate_bbox
)
from llms import conversation as conversation_lib
from utils import (
    AverageMeter, 
    ProgressMeter, 
    Summary, 
    dict_to_cuda, 
    intersectionAndUnionGPU, 
    get_train_transform, 
    get_val_transform, 
    DEFAULT_PC_START_TOKEN, 
    DEFAULT_PC_END_TOKEN
)

import warnings
warnings.filterwarnings("ignore")


def parse_args(args):
    parser = argparse.ArgumentParser(description="PLL Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version", default="Qwen/Qwen2.5-3B" 
    )
    parser.add_argument("--save_pred", action="store_true", default=False)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument(
        "--task_mode",
        default="single",
        type=str,
        choices=["single", "multi", "zero"],
        help="inference mode",
    )
    parser.add_argument("--dataset", default="refer_seg", type=str)
    parser.add_argument("--sample_rates", default="1", type=str)
    parser.add_argument(
        "--sem_seg_data",
        default="scannet200||scannetpp||s3dis",
        type=str,
    )
    parser.add_argument(
        "--scene_cap_data",
        default="scenecap",
        type=str,
    )
    parser.add_argument(
        "--refer_seg_data", default="scanrefer||multi3drefer||nr3d||sr3d", type=str
    )
    parser.add_argument("--refer_sample_rates", default="1", type=str)
    parser.add_argument("--val_dataset", default="refer_seg||scanrefer", type=str)
    parser.add_argument("--dataset_dir", default="./data/plldata", type=str)
    parser.add_argument("--model_max_length", default=2048, type=int)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--log_base_dir", default="./exps", type=str)
    parser.add_argument("--exp_name", default="pll", type=str)
    parser.add_argument("--from_different_models", action="store_true", default=False)
    parser.add_argument("--global_train_step", default=5000, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument(
        "--batch_size", default=1, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=1,
        type=int,
    )
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=10, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=1.0, type=float)
    parser.add_argument("--bce_loss_weight", default=1.0, type=float)
    parser.add_argument("--iou_loss_weight", default=1.0, type=float)
    parser.add_argument("--l1_loss_weight", default=1.0, type=float)
    parser.add_argument("--box_loss_weight", default=1.0, type=float)
    parser.add_argument("--lora_r", default=16, type=int)
    parser.add_argument("--lora_alpha", default=32, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.99, type=float)
    parser.add_argument("--num_prompt_per_sample", default=1, type=int)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="", type=str) 
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_pc_encode", action="store_true", default=False)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument("--OV_instance", action="store_true", default=False)
    parser.add_argument("--OV_detection", action="store_true", default=False)
    parser.add_argument("--OV_recognition", action="store_true", default=False)
    parser.add_argument("--train_pll", action="store_true", default=False)
    parser.add_argument(
        "--conv_type",
        default="qwen",
        type=str,
        choices=["qwen", "llava_v1"],
    )
    parser.add_argument(
        '--vocab_list', 
        type=str,
        nargs='*',
        default=None,
        help="list of vocabularies, default: None"
    )
    parser.add_argument(
        '--valid_idx', 
        type=int,
        nargs='*',
        default=None,
        help="list of vocabularies, default: None"
    )
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    # calculate step
    args.epochs = args.global_train_step // args.steps_per_epoch

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length, 
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    # expand [SEG] token
    num_added_tokens = tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[-1]

    assert args.use_mm_start_end

    model_args = {
        "eval_only": args.eval_only,
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "iou_loss_weight": args.iou_loss_weight,
        "l1_loss_weight": args.l1_loss_weight,
        "box_loss_weight": args.box_loss_weight,
        "seg_token_idx": args.seg_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "use_mm_start_end": args.use_mm_start_end,
        "from_different_models": args.from_different_models,
        "attn_implementation": "eager",
    }
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    model = PLLForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
        **model_args,
        )
    
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    if not args.eval_only:
        if not "pll" in args.version:
            model.get_model().initialize_llm_modules(model.get_model().config)
        if args.train_pll:
            model.get_model().initialize_pll_modules(model.get_model().config)

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    lora_r = args.lora_r
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "point_encoder",
                                "lm_head",
                                "mask_model", 
                                "mlp2feat", 
                                "mlp2llm", 
                                "mlp2ref",
                                "mask_decoder", 
                                "mask_decoder_proj",
                                "mask_iou_head", 
                                "mask_box_head",
                                "mask_pos_emb",
                                "attn_proj", "attn_layer", "attn_norm",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        if args.local_rank == 0:
            model.print_trainable_parameters()

    else:
        for param in model.parameters():
            param.requires_grad = False

    model.resize_token_embeddings(len(tokenizer))

    # make trainable 
    trainable_list = [
        "lm_head", "embed_tokens",
        "mlp2feat", 
        "mlp2llm",
        "mlp2ref",
        "mask_decoder",
        "mask_decoder_proj",
        "mask_iou_head",
        "mask_box_head",
        "mask_pos_emb",
        "attn_proj", "attn_layer", "attn_norm",
        ]
                           
        
    # do not operate on LLM parameters
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in trainable_list
            ]
        ):
            p.requires_grad = True
            if args.local_rank == 0:
                print("Trainable: ", n, "p.shape: ", p.shape)

    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1

    train_dataset = TrainDataset(
        args.dataset_dir,
        tokenizer,
        transform=get_train_transform(),
        precision=args.precision,
        dataset=args.dataset,
        sample_rate=[float(x) for x in args.sample_rates.split(",")],
        sem_seg_data=args.sem_seg_data,
        scene_cap_data=args.scene_cap_data,
        refer_seg_data=args.refer_seg_data,
        refer_sample_rate=args.refer_sample_rates,
        split="train",
    )

    if not args.no_eval:
        val_dataset = ValDataset(
            args.dataset_dir,
            tokenizer,
            transform=get_val_transform(),
            precision=args.precision,
            dataset=args.val_dataset,
            split="val",
            vocab_list=args.vocab_list,
            )
        if args.local_rank == 0:
            print(
                f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples."
            )
    else:
        val_dataset = None
        if args.local_rank == 0:
            print(f"Training with {len(train_dataset)} examples.")

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.global_train_step, # args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0.0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": int(args.warmup_ratio * args.global_train_step),
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }

    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
        ),
        config=ds_config,
    )

    # visual model post init
    model_engine.module.get_model().initialize_mask_model()
    model_engine.param_names = {param: name for name, param in model.named_parameters()}

    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in ["mask_model", "pc_encoder"]
            ]
        ):
            p.requires_grad = False
            # if args.local_rank == 0:
            #     print("Frozen: ", n, "p.shape: ", p.shape)

    if False:
        for n, p in model.named_parameters():
            if args.local_rank == 0:
                print(f"Requires Grad is {p.requires_grad}", n, "p.shape: ", p.shape)
    
    # resume deepspeed checkpoint
    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        print(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
        )

    # validation dataset
    if val_dataset is not None:
        # assert args.val_batch_size == 1
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=args.local_rank,
            ),
        )

    train_iter = iter(train_loader)
    best_score = 0.0

    if args.eval_only:
        miou = validate(val_loader, model_engine, 0, writer, args, tokenizer)
        exit()

    # miou = validate(val_loader, model_engine, 0, writer, args)
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        if args.task_mode == 'zero': 
            train_iter = train_wo_mask(train_loader, model_engine, epoch, scheduler, writer, train_iter, args)
        else:
            train_iter = train(train_loader, model_engine, epoch, scheduler, writer, train_iter, args)

        if args.no_eval == False:
            miou = validate(val_loader, model_engine, epoch, writer, args)
            is_best = miou > best_score
            best_score = max(miou, best_score)

        if args.no_eval or is_best:
            save_dir = os.path.join(args.log_dir, "ckpt_model")
            if args.local_rank == 0:
                torch.save(
                    {"epoch": epoch},
                    os.path.join(
                        args.log_dir,
                        "meta_log_giou{:.3f}.pth".format(best_score),
                    ),
                )
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)


def train_wo_mask(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,

):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":.1f")
    data_time = AverageMeter("Data", ":.1f")
    losses = AverageMeter("Loss", ":.2f")
    ce_losses = AverageMeter("Ce", ":.2f")
    training_lr = AverageMeter("LR", ":.2e")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            ce_losses,
            training_lr,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)

            input_dict = dict_to_cuda(input_dict)
            input_dict["precision"] = args.precision
            input_dict["task_mode"] = args.task_mode

            output_dict = model(**input_dict)

            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]

            losses.update(loss.item(), len(input_dict["points_coord"]))
            ce_losses.update(ce_loss.item(), len(input_dict["points_coord"]))
            model.backward(loss)
            model.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()

                losses.all_reduce()
                ce_losses.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, global_step)
                writer.add_scalar("train/ce_loss", ce_losses.avg, global_step)
                writer.add_scalar(
                    "metrics/total_secs_per_batch", batch_time.avg, global_step
                )
                writer.add_scalar(
                    "metrics/data_secs_per_batch", data_time.avg, global_step
                )

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            training_lr.update(curr_lr[0])
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

    return train_iter


def train(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,

):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":.1f")
    data_time = AverageMeter("Data", ":.1f")
    losses = AverageMeter("Loss", ":.2f")
    ce_losses = AverageMeter("Ce", ":.2f")
    mask_bce_losses = AverageMeter("BCE", ":.2f")
    mask_dice_losses = AverageMeter("DICE", ":.2f")
    iou_losses = AverageMeter("IOU", ":.2f")
    l1_losses = AverageMeter("L1", ":.2f")
    bbox_losses = AverageMeter("BOX", ":.2f")
    training_iou = AverageMeter("PRED_IOU", ":.4f")
    positive_pred = AverageMeter("POS", ":.4f")
    training_lr = AverageMeter("LR", ":.2e")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            ce_losses,
            mask_bce_losses,
            mask_dice_losses,
            iou_losses,
            l1_losses,
            bbox_losses,
            training_iou,
            positive_pred,
            training_lr,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)

            input_dict = dict_to_cuda(input_dict)
            input_dict["precision"] = args.precision
            input_dict["task_mode"] = args.task_mode

            output_dict = model(**input_dict)

            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            mask_bce_loss = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]
            iou_loss = output_dict["iou_loss"]
            l1_loss = output_dict["l1_loss"]
            bbox_loss = output_dict["bbox_loss"]

            pred_masks = output_dict["pred_masks"]
            gt_masks = output_dict["gt_masks"]
            gt_masks = [mask.int() for mask in gt_masks]
            
            acc_iou, pos_pred = 0.0, 0.0
            for i, (mask_i, pred_i) in enumerate(zip(gt_masks, pred_masks)):
                p_masks = torch.sigmoid(pred_i)
                pred_mask = (p_masks > 0.5).int()
                # pred_mask = pred_mask[:, (pred_mask > 0).sum(dim=0) >= 150]
                pred_mask = pred_mask.any(dim=1).int().view(1, -1)

                pos_pred = pred_mask.sum() / pred_i.shape[0]
                intersection_i, union_i, _ = intersectionAndUnionGPU(
                    pred_mask.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=-1
                )
                acc_iou += intersection_i / (union_i + 1e-6)
                acc_iou[union_i == 0] += 1.0  # no-object target
            acc_iou = acc_iou / len(pred_masks)

            training_iou.update(acc_iou[1].item(), len(input_dict["points_coord"]))
            positive_pred.update(pos_pred.item(), len(input_dict["points_coord"]))

            losses.update(loss.item(), len(input_dict["points_coord"]))
            ce_losses.update(ce_loss.item(), len(input_dict["points_coord"]))
            mask_bce_losses.update(mask_bce_loss.item(), len(input_dict["points_coord"]))
            mask_dice_losses.update(mask_dice_loss.item(), len(input_dict["points_coord"]))
            iou_losses.update(iou_loss.item(), len(input_dict["points_coord"]))
            l1_losses.update(l1_loss.item(), len(input_dict["points_coord"]))
            bbox_losses.update(bbox_loss.item(), len(input_dict["points_coord"]))
            model.backward(loss)
            model.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()

                losses.all_reduce()
                ce_losses.all_reduce()
                mask_bce_losses.all_reduce()
                mask_dice_losses.all_reduce()
                iou_losses.all_reduce()
                l1_losses.all_reduce()
                bbox_losses.all_reduce()
                training_iou.all_reduce()
                positive_pred.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, global_step)
                writer.add_scalar("train/ce_loss", ce_losses.avg, global_step)
                writer.add_scalar(
                    "train/mask_bce_loss", mask_bce_losses.avg, global_step
                )
                writer.add_scalar(
                    "train/mask_dice_loss", mask_dice_losses.avg, global_step
                )
                writer.add_scalar("train/iou_losses", iou_losses.avg, global_step)
                writer.add_scalar("train/l1_losses", l1_losses.avg, global_step)
                writer.add_scalar("train/bbox_loss", bbox_losses.avg, global_step)
                writer.add_scalar(
                    "metrics/total_secs_per_batch", batch_time.avg, global_step
                )
                writer.add_scalar(
                    "metrics/data_secs_per_batch", data_time.avg, global_step
                )

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            mask_bce_losses.reset()
            mask_dice_losses.reset()
            iou_losses.reset()
            l1_losses.reset()
            bbox_losses.reset()
            training_iou.reset()
            positive_pred.reset()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            training_lr.update(curr_lr[0])
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

    return train_iter


def validate(val_loader, model_engine, steps, writer, args, tokenizer=None):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    model_engine.eval()

    if args.save_pred:
        vis_save_path = os.path.join(args.log_dir, "vis_output")

    precision_25_count, precision_50_count = 0, 0
    total_samples = 0

    predict_scene_results = {}

    for input_dict in tqdm.tqdm(val_loader):

        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        input_dict["precision"] = args.precision
        input_dict["task_mode"] = args.task_mode

        with torch.no_grad():
            output_dict = model_engine(**input_dict)

        pred_index = input_dict["points_index"]
        pred_masks = output_dict["pred_masks"]
        gt_masks = output_dict["gt_masks"]
        gt_masks = [mask.int() for mask in gt_masks]
        valid_gt = [mask.sum() > 0 for mask in gt_masks]

        # ov instance
        mask_lists = output_dict["mask_list"]

        # assert len(pred_masks) == 1

        intersection, union, acc_iou = 0.0, 0.0, 0.0
        for i, (mask_i, pred_i) in enumerate(zip(gt_masks, pred_masks)):
            p_masks = torch.sigmoid(pred_i)
            pred_mask_ = (p_masks > 0.5).int()
            # pred_mask = pred_mask[:, (pred_mask > 0).sum(dim=0) >= 150]
            pred_mask = pred_mask_.any(dim=1).int().view(1, -1)
            
            mask_list = torch.sigmoid(mask_lists[i])
            mask_list = (mask_list > 0.5).int()

            predict_scene_results = assign_semantic_to_instance(
                input_dict['scene_paths'][i], 
                input_dict['class_id_list'][i], 
                mask_list, 
                pred_mask_, 
                predict_scene_results
                )
            
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                pred_mask.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=-1
            )

            intersection += intersection_i
            union += union_i
            acc_iou_ = intersection_i / (union_i + 1e-6)
            acc_iou += acc_iou_

            acc_iou[union_i == 0] += 1.0  # no-object target

            # Calculate precision thresholds
            if acc_iou_[1] > 0.25:
                precision_25_count += 1
            if acc_iou_[1] > 0.5:
                precision_50_count += 1
            total_samples += 1

        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / len(gt_masks)
        acc_iou_meter.update(acc_iou, n=len(gt_masks))
        intersection_meter.update(intersection), union_meter.update(union)

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    miou = acc_iou_meter.avg[1]
    precision_25 = precision_25_count / total_samples
    precision_50 = precision_50_count / total_samples

    if args.OV_instance or args.OV_detection:
        predict_scene_results = get_valid_results(predict_scene_results)
        dataset_name = args.val_dataset.split("||")[-1]  # "scannet" 
        class_label = [re.sub(r'_', ' ', vocab) for vocab in args.vocab_list]
        valid_idx = np.array(args.valid_idx)
        gt_path = os.path.join(args.dataset_dir, "openins", dataset_name, "ground_truth")
        if args.OV_instance:
            evaluate_inst(predict_scene_results, gt_path, class_label, valid_idx)

        if args.OV_detection:
            scene_root = "scannet" if "scannet" in dataset_name else dataset_name
            scene_path = os.path.join(args.dataset_dir, scene_root)
            evaluate_bbox(predict_scene_results, gt_path, scene_path, class_label, valid_idx)

    if args.local_rank == 0:
        writer.add_scalar("val/miou", miou, steps)
        writer.add_scalar("val/precision_25", precision_25, steps)
        writer.add_scalar("val/precision_50", precision_50, steps)
        print("miou: {:.4f}, acc@0.25: {:.4f}, acc@0.5: {:.4f}".format(miou, precision_25, precision_50))

    return miou


def pairwise_iou(masks1: torch.Tensor, masks2: torch.Tensor, eps: float = 1e-6):
    intersection = masks1.T @ masks2  # (M1, M2)
    union = masks1.sum(0)[:, None] + masks2.sum(0)[None, :] - intersection  # M1, M2
    iou = intersection / (union + eps)
    return iou


def assign_semantic_to_instance(
    scene_name,
    class_id,
    mask_list,
    pred_mask,
    predict_scene_results,
    iou_threshold=0.1,
):
    if scene_name not in predict_scene_results:
        predict_scene_results[scene_name] = {
            "pred_masks": mask_list,
            "history": [[] for _ in range(mask_list.shape[1])],
        }

    scene_result = predict_scene_results[scene_name]
    mask_list = scene_result["pred_masks"]

    iou = pairwise_iou(mask_list.float(), pred_mask.float()).max(-1)[0]

    for idx in range(mask_list.shape[1]):
        if iou[idx] > iou_threshold:
            scene_result["history"][idx].append({
                "iou": iou[idx].item(),
                "class_id": class_id,
            })

    return predict_scene_results

def get_valid_results(predict_scene_results):
    for scene_name, scene_result in predict_scene_results.items():
        history = scene_result["history"]
        num_instances = len(history)

        pred_scores = torch.zeros(num_instances, dtype=torch.float32)
        pred_classes = torch.ones(num_instances, dtype=torch.float32) * -1

        for idx, records in enumerate(history):
            if len(records) == 0:
                continue

            class_iou_map = {}
            for record in records:
                class_id = record["class_id"]
                iou = record["iou"]
                if class_id not in class_iou_map:
                    class_iou_map[class_id] = []
                class_iou_map[class_id].append(iou)

            class_iou_sum = {clss: sum(ious) for clss, ious in class_iou_map.items()}

            best_class = max(class_iou_sum, key=class_iou_sum.get)
            best_class_iou_sum = class_iou_sum[best_class]
            best_class_iou_mean = best_class_iou_sum / len(class_iou_map[best_class])

            pred_classes[idx] = best_class
            pred_scores[idx] = best_class_iou_mean

        valid_mask = pred_classes != -1

        scene_result["pred_masks"] = scene_result["pred_masks"][:, valid_mask].cpu()
        scene_result["pred_scores"] = pred_scores[valid_mask].float().cpu()
        scene_result["pred_classes"] = pred_classes[valid_mask].cpu()

    return predict_scene_results


if __name__ == "__main__":
    main(sys.argv[1:])