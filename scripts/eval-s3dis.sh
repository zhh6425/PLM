export PYTHONPATH=./

CUDA_VISIBLE_DEVICES="0" deepspeed --master_port=40000 pll/main.py \
  --version="exps/pll-llama-7b-3e-4/hf_model" \
  --dataset_dir='./data/plldata' \
  --vision_pretrained="pll/ckpt/mask3d/scannet200.ckpt" \
  --val_dataset "sem_seg||s3dis" \
  --num_prompt_per_sample 1 \
  --val_batch_size 8 \
  --exp_name="pll-eval-s3dis" \
  --conv_type "llava_v1" \
  --eval_only \
  --task_mode "multi" \
  --OV_instance \
  --vocab_list floor window door table bookcase board \
  --valid_idx 1 5 6 7 10 11 \

# - / 12
# 0 1 2 3 4 5 6 7 8 9 10 11
# ceiling floor wall beam column window door table chair sofa bookcase board

# - / 6
# 1 5 6 7 10 11
# floor window door table bookcase board

# - / 4
# 5 7 9 10
# window table sofa bookcase 