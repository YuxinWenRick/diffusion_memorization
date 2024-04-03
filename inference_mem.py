import argparse
import wandb
import copy
from tqdm import tqdm
from statistics import mean
from PIL import Image

import torch

import open_clip
from optim_utils import *
from io_utils import *

from local_sd_pipeline import LocalStableDiffusionPipeline
from diffusers import DDIMScheduler, UNet2DConditionModel


def main(args):
    table = None
    if args.with_tracking:
        wandb.init(
            project="diffusion_memorization", name=args.run_name, tags=["run_mem"]
        )
        wandb.config.update(args)
        table = wandb.Table(
            columns=[
                "gt_prompt",
                "gen_prompt",
                "gt_clip_score",
                "gen_clip_score",
                "SSCD_sim",
                "SSCD_sim_max",
                "SSCD_sim_min",
            ]
        )

    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.unet_id is not None:
        unet = UNet2DConditionModel.from_pretrained(
            args.unet_id, torch_dtype=torch.bfloat16
        )
        pipe = LocalStableDiffusionPipeline.from_pretrained(
            args.model_id,
            unet=unet,
            torch_dtype=torch.bfloat16,
            safety_checker=None,
            requires_safety_checker=False,
        )
    else:
        pipe = LocalStableDiffusionPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            safety_checker=None,
            requires_safety_checker=False,
        )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    # dataset
    set_random_seed(args.gen_seed)
    dataset, prompt_key = get_dataset_finetune(args.dataset)

    args.end = min(args.end, len(dataset))

    # generation
    print("generation")
    all_gen_images = []
    all_gt_images = []
    all_gen_prompts = []
    all_gt_prompts = []

    for i in tqdm(range(args.start, args.end)):
        seed = i + args.gen_seed

        gt_prompt = dataset[i][prompt_key]

        ### prompt modification
        if args.prompt_aug_style is not None:
            prompt = prompt_augmentation(
                gt_prompt,
                args.prompt_aug_style,
                tokenizer=pipe.tokenizer,
                repeat_num=args.repeat_num,
            )
        else:
            prompt = gt_prompt

        ### optim prompt
        if args.optim_target_loss is not None:
            set_random_seed(seed)
            auged_prompt_embeds = pipe.aug_prompt(
                prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=args.num_images_per_prompt,
                target_steps=[args.optim_target_steps],
                lr=args.optim_lr,
                optim_iters=args.optim_iters,
                target_loss=args.optim_target_loss,
            )

            ### generation
            set_random_seed(seed)
            outputs = pipe(
                prompt_embeds=auged_prompt_embeds,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=args.num_images_per_prompt,
            )
        else:
            outputs = pipe(
                prompt=prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=args.num_images_per_prompt,
            )

        gen_images = outputs.images

        if "groundtruth" in args.dataset:
            gt_images = []

            curr_index = dataset[i]["index"]
            for filename in glob.glob(f"{args.dataset}/gt_images/{curr_index}/*.png"):
                im = Image.open(filename)
                gt_images.append(im)
        else:
            gt_images = [dataset[i]["image"]]

        all_gen_images.append(gen_images)
        all_gt_images.append(gt_images)
        all_gen_prompts.append(prompt)
        all_gt_prompts.append(gt_prompt)

    pipe = pipe.to(torch.device("cpu"))
    del pipe
    if "pez_model" in args:
        pez_model = args.pez_model.to(torch.device("cpu"))
        del pez_model
        del args.pez_model
    torch.cuda.empty_cache()

    # similarity model
    sim_model = torch.jit.load("sscd_disc_large.torchscript.pt").to(device)

    # reference model
    if args.reference_model is not None:
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(
            args.reference_model,
            pretrained=args.reference_model_pretrain,
            device=device,
        )
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

    # eval
    print("eval")
    gt_clip_scores = []
    gen_clip_scores = []
    SSCD_sims = []
    SSCD_sims_max = []
    SSCD_sims_min = []

    for i in tqdm(range(len(all_gen_images))):
        gen_images = all_gen_images[i]
        gt_images = all_gt_images[i]
        prompt = all_gen_prompts[i]
        gt_prompt = all_gt_prompts[i]

        ### SSCD sim
        SSCD_sim = measure_SSCD_similarity(gt_images, gen_images, sim_model, device)
        gt_image = gt_images[SSCD_sim.argmax(dim=0)[0].item()]
        SSCD_sim = SSCD_sim.max(0).values
        SSCD_sim_max = SSCD_sim.max().item()
        SSCD_sim_min = SSCD_sim.min().item()
        SSCD_sim = SSCD_sim.mean().item()

        SSCD_sims.append(SSCD_sim)
        SSCD_sims_max.append(SSCD_sim_max)
        SSCD_sims_min.append(SSCD_sim_min)

        ### clip score
        if args.reference_model is not None:
            sims = measure_CLIP_similarity(
                [gt_image] + gen_images,
                gt_prompt,
                ref_model,
                ref_clip_preprocess,
                ref_tokenizer,
                device,
            )
            gt_clip_score = sims[0:1].mean().item()
            gen_clip_score = sims[1:].mean().item()
        else:
            gt_clip_score = 0
            gen_clip_score = 0

        gt_clip_scores.append(gt_clip_score)
        gen_clip_scores.append(gen_clip_score)

        if args.with_tracking:
            table.add_data(
                gt_prompt,
                prompt,
                gt_clip_score,
                gen_clip_score,
                SSCD_sim,
                SSCD_sim_max,
                SSCD_sim_min,
            )

    if args.with_tracking:
        wandb.log({"Table": table})
        wandb.log(
            {
                "gt_clip_score_mean": mean(gt_clip_scores),
                "gen_clip_score_mean": mean(gen_clip_scores),
                "SSCD_sim_mean": mean(SSCD_sims),
                "SSCD_sim_max_mean": mean(SSCD_sims_max),
                "SSCD_sim_min_mean": mean(SSCD_sims_min),
            }
        )

    print(f"gt_clip_score_mean: {mean(gt_clip_scores)}")
    print(f"gen_clip_score_mean: {mean(gen_clip_scores)}")
    print(f"SSCD_sim_mean: {mean(SSCD_sims)}")
    print(f"SSCD_sim_max_mean: {mean(SSCD_sims_max)}")
    print(f"SSCD_sim_min_mean: {mean(SSCD_sims_min)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="diffusion memorization")
    parser.add_argument("--run_name", default="test")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=500, type=int)
    parser.add_argument("--image_length", default=512, type=int)
    parser.add_argument("--model_id", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--unet_id", default=None)
    parser.add_argument("--with_tracking", action="store_true")
    parser.add_argument("--num_images_per_prompt", default=4, type=int)
    parser.add_argument("--guidance_scale", default=7.5, type=float)
    parser.add_argument("--num_inference_steps", default=50, type=int)
    parser.add_argument("--reference_model", default=None)
    parser.add_argument("--reference_model_pretrain", default="laion2b_s12b_b42k")
    parser.add_argument("--gen_seed", default=0, type=int)

    # mitigation strategy
    # baseline
    parser.add_argument(
        "--prompt_aug_style", default=None
    )  # rand_numb_add, rand_word_add, rand_word_repeat
    parser.add_argument("--repeat_num", default=1, type=int)

    # ours
    parser.add_argument("--optim_target_steps", default=0, type=int)
    parser.add_argument("--optim_lr", default=0.05, type=float)
    parser.add_argument("--optim_iters", default=10, type=int)
    parser.add_argument("--optim_target_loss", default=None, type=float)

    args = parser.parse_args()

    main(args)
