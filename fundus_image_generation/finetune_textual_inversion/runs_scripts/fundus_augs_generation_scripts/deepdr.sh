cd ../../dr_aug \
# class 0
python generate_fundus_augmentations.py \
--out="./generated_fundus_aug/deepdr/0" \
--model-path="../finetune_dreambooth/train_outputs_checkpoints/grade_0" \
--embed-path="./train_outputs_checkpoints/fundus-tokens/fundus-3-16.pt" \
--dataset="fundus_dr_generic" \
--data_dir="./datasets/images/deepdr" \
--dr_grade=0 \
--seed=0 \
--examples-per-class=714 \
--num-synthetic=10 \
--prompt="a photo of fundus" \
--aug="textual-inversion" \
--guidance-scale=7.5 \
--strength=0.5 \
--mask=0 \
--inverted=0 \
--compose="parallel" \
--class_name="no diabetic retinopathy" \

# class 1
python generate_fundus_augmentations.py \
--out="./generated_fundus_aug/deepdr/1" \
--model-path="../finetune_dreambooth/train_outputs_checkpoints/grade_1" \
--embed-path="./train_outputs_checkpoints/fundus-tokens/fundus-3-16.pt" \
--dataset="fundus_dr_generic" \
--data_dir="./datasets/images/deepdr" \
--dr_grade=1 \
--seed=0 \
--examples-per-class=186 \
--num-synthetic=10 \
--prompt="a photo of fundus" \
--aug="textual-inversion" \
--guidance-scale=7.5 \
--strength=0.5 \
--mask=0 \
--inverted=0 \
--compose="parallel" \
--class_name="mild diabetic retinopathy" \

# class 2
python generate_fundus_augmentations.py \
--out="./generated_fundus_aug/deepdr/2" \
--model-path="../finetune_dreambooth/train_outputs_checkpoints/grade_2" \
--embed-path="./train_outputs_checkpoints/fundus-tokens/fundus-3-16.pt" \
--dataset="fundus_dr_generic" \
--data_dir="./datasets/images/deepdr" \
--dr_grade=2 \
--seed=0 \
--examples-per-class=326 \
--num-synthetic=10 \
--prompt="a photo of fundus" \
--aug="textual-inversion" \
--guidance-scale=7.5 \
--strength=0.5 \
--mask=0 \
--inverted=0 \
--compose="parallel" \
--class_name="moderate diabetic retinopathy" \

# class 3
python generate_fundus_augmentations.py \
--out="./generated_fundus_aug/deepdr/3" \
--model-path="../finetune_dreambooth/train_outputs_checkpoints/grade_3" \
--embed-path="./train_outputs_checkpoints/fundus-tokens/fundus-3-16.pt" \
--dataset="fundus_dr_generic" \
--data_dir="./datasets/images/deepdr" \
--dr_grade=3 \
--seed=0 \
--examples-per-class=282 \
--num-synthetic=10 \
--prompt="a photo of fundus" \
--aug="textual-inversion" \
--guidance-scale=7.5 \
--strength=0.5 \
--mask=0 \
--inverted=0 \
--compose="parallel" \
--class_name="severe diabetic retinopathy" \


# class 4
python generate_fundus_augmentations.py \
--out="./generated_fundus_aug/deepdr/4" \
--model-path="../finetune_dreambooth/train_outputs_checkpoints/grade_4" \
--embed-path="./train_outputs_checkpoints/fundus-tokens/fundus-3-16.pt" \
--dataset="fundus_dr_generic" \
--data_dir="./datasets/images/deepdr" \
--dr_grade=4 \
--seed=0 \
--examples-per-class=92 \
--num-synthetic=10 \
--prompt="a photo of fundus" \
--aug="textual-inversion" \
--guidance-scale=7.5 \
--strength=0.5 \
--mask=0 \
--inverted=0 \
--compose="parallel" \
--class_name="proliferative diabetic retinopathy" \

