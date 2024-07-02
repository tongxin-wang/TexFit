export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export dataset_name="dataset/dfmm_spotlight_hf"
export CUDA_VISIBLE_DEVICES="0"

python train_texfit.py \
  --mixed_precision="fp16" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --use_ema \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=140000 \
  --checkpointing_steps=20000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-model-finetuned/texfit-model"