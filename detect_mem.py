import argparse
from tqdm import tqdm

import torch

from optim_utils import *
from io_utils import *

from local_sd_pipeline import LocalStableDiffusionPipeline
from diffusers import DDIMScheduler, UNet2DConditionModel


def main(args):
    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.unet_id is not None:
        unet = UNet2DConditionModel.from_pretrained(
            args.unet_id, torch_dtype=torch.float16
        )
        pipe = LocalStableDiffusionPipeline.from_pretrained(
            args.model_id,
            unet=unet,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        )
    else:
        pipe = LocalStableDiffusionPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    # dataset
    set_random_seed(args.gen_seed)
    dataset, prompt_key = get_dataset(args.dataset, pipe)

    args.end = min(args.end, len(dataset))

    # generation
    print("generation")

    all_metrics = ["uncond_noise_norm", "text_noise_norm"]
    all_tracks = []

    for i in tqdm(range(args.start, args.end)):
        seed = i + args.gen_seed

        prompt = dataset[i][prompt_key]

        ### generation
        set_random_seed(seed)
        outputs, track_stats = pipe(
            prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=args.num_images_per_prompt,
            track_noise_norm=True,
        )

        uncond_noise_norm, text_noise_norm = (
            track_stats["uncond_noise_norm"],
            track_stats["text_noise_norm"],
        )

        curr_line = {}
        for metric_i in all_metrics:
            values = locals()[metric_i]
            curr_line[f"{metric_i}"] = values

        curr_line["prompt"] = prompt

        all_tracks.append(curr_line)

    os.makedirs("det_outputs", exist_ok=True)
    write_jsonlines(all_tracks, f"det_outputs/{args.run_name}.jsonl")


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
    parser.add_argument("--gen_seed", default=0, type=int)

    args = parser.parse_args()

    main(args)
