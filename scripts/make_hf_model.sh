export PYTHONPATH=./

CUDA_VISIBLE_DEVICES="" python pll/merge_lora_weight_to_huggingface_model.py \
  --version="meta-llama/Llama-2-7b-hf" \
  --vision_pretrained="pll/ckpt/mask3d/scannet200.ckpt" \
  --lora_r 8 \
  --lora_alpha 16 \
  --conv_type "llava_v1" \
  --precision "bf16" \
  --weight="exps/pll-llama-7b-3e-4/pytorch_model.bin" \
  --save_path="exps/pll-llama-7b-3e-4/hf_model" \
  --train_pll \