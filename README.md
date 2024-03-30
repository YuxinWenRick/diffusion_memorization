# Detecting, Explaining, and Mitigating Memorization in Diffusion Models
Official repo for [Detecting, Explaining, and Mitigating Memorization in Diffusion Models](https://openreview.net/forum?id=84n3UwkH7b).

If you have any questions, feel free to email Yuxin (<ywen@umd.edu>).

## Dependencies
- PyTorch == 1.13.0
- transformers == 4.30.2
- diffusers == 0.18.2
- accelerate == 0.21.0
- datasets

## Play around our methods
We provide detecting, explaining, and (inference) mitigating methods separately in the Jupyter notebooks within the `examples/` folder. You can directly interact with our methods without the need to download any datasets or fine-tune any models.

## To reproduce the results in our paper
### Detect memorization
1. Run detection:

memorized prompts
```
python detect_mem.py --run_name memorized_prompts --dataset examples/sdv1_500_memorized.jsonl --end 500 --gen_seed 0
```

non-memorized prompts
```
python detect_mem.py --run_name non_memorized_prompts --dataset Gustavosta/Stable-Diffusion-Prompts --end 500 --gen_seed 0
```

2. Visualize resuts: you could use `examples/det_mem_viz.ipynb`.

### Explain memorization
You could check out `examples/token_wise_significance.ipynb`.

### Mitigate memorization
For our mitigation experiments, we fine-tune stable diffusion models on LAION to experiment with memorized examples with more meaningful prompts.

You could either use our fine-tuned models by downloading the pre-fine-tuned [checkpoint](https://drive.google.com/drive/folders/1XiYtYySpTUmS_9OwojNo4rsPbkfCQKBl?usp=sharing) and [memorized images](https://drive.google.com/drive/folders/1oQ49pO9gwwMNurxxVw7jwlqHswzj6Xbd?usp=sharing) or fine-tune the model with the following instruction:

### Fine-tune Stable Diffusion model
```
accelerate launch --mixed_precision=fp16 train_text_to_image.py --dataset=$MEM_DATA --non_mem_dataset=$NON_MEM_DATA --output_dir=finetuned_checkpoints/v1-20000 --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4 --end=200 --repeats=200 --non_mem_ratio=3 --use_ema --resolution=512 --center_crop --train_batch_size=8 --gradient_accumulation_steps=1 --gradient_checkpointing --max_train_steps=20000 --checkpointing_steps=20000 --learning_rate=1e-05 --max_grad_norm=1 --lr_scheduler=constant --lr_warmup_steps=0
```

`dataset` and `non_mem_dataset` should provide the path to the dataset you want to memorize and the base dataset respectively.

By default, the code assumes the data folder that contains the image and text pairs looks like:
```
-- daataset
    |-- 00000001.jpg
    |-- 00000001.txt
    |-- 00000002.jpg
    |-- 00000002.txt
    |-- ...
```

### Inference-time mititgation
You may want to download the SSCD checkpoint first [here](https://drive.google.com/file/d/1PAMwyK5b5zi6WBvyENtWuWr0lpT-TYMk/view?usp=sharing).

No mitigation:
```
python inference_mem.py --run_name no_mitigation --unet_id finetuned_checkpoints/checkpoint-20000/unet --dataset memorized_images --end 200 --gen_seed 0 --reference_model ViT-g-14 --with_tracking
```

Random token addition baseline:
```
python inference_mem.py --run_name add_rand_word --unet_id finetuned_checkpoints/checkpoint-20000/unet --dataset memorized_images --end 200 --gen_seed 0 --prompt_aug_style rand_word_add --repeat_num 4 --reference_model ViT-g-14 --with_tracking
```

Ours:
```
python inference_mem.py --run_name ours --unet_id finetuned_checkpoints/checkpoint-20000/unet --dataset memorized_images --end 200 --gen_seed 0 --optim_target_loss 3 --optim_target_steps 0 --reference_model ViT-g-14 --with_tracking
```

### Training-time mitigation
No mitigation:
```
accelerate launch --mixed_precision=fp16 train_text_to_image.py --dataset=$MEM_DATA --non_mem_dataset=$NON_MEM_DATA --output_dir=finetuned_checkpoints/no_mitigation --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4 --end=200 --repeats=200 --non_mem_ratio=3 --use_ema --resolution=512 --center_crop --train_batch_size=8 --gradient_accumulation_steps=1 --gradient_checkpointing --max_train_steps=20000 --checkpointing_steps=20000 --learning_rate=1e-05 --max_grad_norm=1 --lr_scheduler=constant --lr_warmup_steps=0
```

Random token addition baseline:
```
accelerate launch --mixed_precision=fp16 train_text_to_image.py --dataset=$MEM_DATA --non_mem_dataset=$NON_MEM_DATA --output_dir=finetuned_checkpoints/add_rand_word --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4 --end=200 --repeats=200 --non_mem_ratio=3 --use_ema --resolution=512 --center_crop --train_batch_size=8 --gradient_accumulation_steps=1 --gradient_checkpointing --max_train_steps=20000 --checkpointing_steps=20000 --learning_rate=1e-05 --max_grad_norm=1 --lr_scheduler=constant --lr_warmup_steps=0 --prompt_aug_style rand_word_add --repeat_num 1
```

Ours:
```
accelerate launch --mixed_precision=fp16 train_text_to_image.py --dataset=$MEM_DATA --non_mem_dataset=$NON_MEM_DATA --output_dir=finetuned_checkpoints/ours --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4 --end=200 --repeats=200 --non_mem_ratio=3 --use_ema --resolution=512 --center_crop --train_batch_size=8 --gradient_accumulation_steps=1 --gradient_checkpointing --max_train_steps=20000 --checkpointing_steps=20000 --learning_rate=1e-05 --max_grad_norm=1 --lr_scheduler=constant --lr_warmup_steps=0 --hard_threshold 2
```

Then, you can use the fnference-time mititgation script to evaluate the memorization:
```
python run_mem.py --run_name baseline --unet_id finetuned_checkpoints/checkpoint-20000/unet --dataset memorized_images --end 200 --gen_seed 0 --reference_model ViT-g-14 --with_tracking
```

## Suggestions and pull requests are welcome!

## Cite our work
If you find this work useful, please cite our paper:

```bibtex
@inproceedings{
wen2024detecting,
title={Detecting, Explaining, and Mitigating Memorization in Diffusion Models},
author={Yuxin Wen and Yuchen Liu and Chen Chen and Lingjuan Lyu},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=84n3UwkH7b}
}
```
