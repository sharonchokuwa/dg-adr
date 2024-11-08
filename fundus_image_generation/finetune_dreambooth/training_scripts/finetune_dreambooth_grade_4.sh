cd ../diffusers/examples/dreambooth
python train_dreambooth.py \
  --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
  --train_text_encoder \
  --revision=fp16 \
  --instance_data_dir=datasets/eyepacs/instance_images/4 \
  --class_data_dir=datasets/eyepacs/class_images/4 \
  --output_dir=../train_outputs_checkpoints/grade_4 \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of fundus with proliferative diabetic retinopathy" \
  --class_prompt="a photo of fundus" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --use_8bit_adam \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=500 \
  --max_train_steps=30000 \
  --mixed_precision=fp16 \
  --seed=0 \
  --checkpointing_steps=15000 \