export PYTHONPATH=./

CUDA_VISIBLE_DEVICES="0" deepspeed --master_port=40000 pll/main.py \
  --version="exps/pll-llama-7b-3e-4/hf_model" \
  --dataset_dir='./data/plldata' \
  --vision_pretrained="pll/ckpt/mask3d/scannet200.ckpt" \
  --val_dataset "sem_seg||scannet20" \
  --num_prompt_per_sample 1 \
  --val_batch_size 8 \
  --exp_name="pll-eval-scannet20" \
  --conv_type "llava_v1" \
  --eval_only \
  --task_mode "multi" \
  --OV_instance \
  --vocab_list bookshelf picture counter desk refridgerator shower_curtain toilet sink bathtub \
  --valid_idx 9 10 11 12 14 15 16 17 18 \

# - / 17
# 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18
# cabinet bed chair sofa table door window bookshelf picture counter desk curtain refridgerator shower_curtain toilet sink bathtub

# - / 7
# 3 4 6 9 10 17 18
# bed chair table bookshelf picture sink bathtub

# - / 9
# 9 10 11 12 14 15 16 17 18
# bookshelf picture counter desk refridgerator shower_curtain toilet sink bathtub

# - / 4
# 5 9 12 16
# sofa bookshelf desk toilet  
