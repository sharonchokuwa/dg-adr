cd ../dr_aug
python fine_tune.py \
--save_steps=500 \
--dataset=fundus \
--output_dir="/outputs_path" \
--resolution=512 \
--train_batch_size=1 \
--lr_warmup_steps=0 \
--gradient_accumulation_steps=1 \
--max_train_steps=1000 \
--learning_rate=5.0e-04 \
--scale_lr \
--lr_scheduler="constant" \
--mixed_precision=fp16 \
--revision=fp16 \
--gradient_checkpointing \
--only_save_embeds \
--num-trials=4 \
--examples-per-class 1 2 4 8 16 \
--checkpointing_steps=10000 \
