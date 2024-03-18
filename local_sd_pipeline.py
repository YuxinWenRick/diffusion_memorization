import torch

from diffusers import StableDiffusionPipeline
from diffusers.utils import randn_tensor
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput


# credit: https://stackoverflow.com/questions/67370107/how-to-sample-similar-vectors-given-a-vector-and-cosine-similarity-in-pytorch
def torch_cos_sim(v, cos_theta, n_vectors=1, EXACT=True):
    """
    EXACT - if True, all vectors will have exactly cos_theta similarity.
            if False, all vectors will have >= cos_theta similarity
    v - original vector (1D tensor)
    cos_theta -cos similarity in range [-1,1]
    """

    # unit vector in direction of v
    u = v / torch.norm(v)
    u = u.unsqueeze(0).repeat(n_vectors, 1)

    # random vector with elements in range [-1,1]
    r = (torch.rand([n_vectors, len(v)]) * 2 - 1).to(v.device).to(v.dtype)

    # unit vector perpendicular to v and u
    uperp = torch.stack([r[i] - (torch.dot(r[i], u[i]) * u[i]) for i in range(len(u))])
    uperp = uperp / (torch.norm(uperp, dim=1).unsqueeze(1).repeat(1, v.shape[0]))

    if not EXACT:
        cos_theta = torch.rand(n_vectors) * (1 - cos_theta) + cos_theta
        cos_theta = cos_theta.unsqueeze(1).repeat(1, v.shape[0])

    # w is the linear combination of u and uperp with coefficients costheta
    # and sin(theta) = sqrt(1 - costheta**2), respectively:
    w = cos_theta * u + torch.sqrt(1 - torch.tensor(cos_theta) ** 2) * uperp

    return w


class LocalStableDiffusionPipeline(StableDiffusionPipeline):
    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        safety_checker,
        feature_extractor,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=requires_safety_checker,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        height=None,
        width=None,
        num_inference_steps=50,
        guidance_scale=7.5,
        negative_prompt=None,
        num_images_per_prompt=1,
        eta=0.0,
        generator=None,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        output_type="pil",
        return_dict=True,
        callback=None,
        callback_steps=1,
        cross_attention_kwargs=None,
        track_noise_norm=False,
        lp=2,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        # self.check_inputs(
        #     prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        # )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        if track_noise_norm is True:
            uncond_noise_norm = []
            text_noise_norm = []

            for i in range(len(latents)):
                uncond_noise_norm.append([])
                text_noise_norm.append([])

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

                    noise_pred_text = noise_pred_text - noise_pred_uncond

                    noise_pred = noise_pred_uncond + guidance_scale * noise_pred_text

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

                if track_noise_norm is True:
                    for j in range(len(uncond_noise_norm)):
                        uncond_noise_norm[j].append(
                            noise_pred_uncond[j].norm(p=lp).item()
                        )
                        text_noise_norm[j].append(noise_pred_text[j].norm(p=lp).item())

        if not output_type == "latent":
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False
            )[0]
            image, has_nsfw_concept = self.run_safety_checker(
                image, device, prompt_embeds.dtype
            )
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        if track_noise_norm is True:
            track_stats = {
                "uncond_noise_norm": uncond_noise_norm,
                "text_noise_norm": text_noise_norm,
            }
            return (
                StableDiffusionPipelineOutput(
                    images=image, nsfw_content_detected=has_nsfw_concept
                ),
                track_stats,
            )
        else:
            return StableDiffusionPipelineOutput(
                images=image, nsfw_content_detected=has_nsfw_concept
            )

    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start

    def prepare_latents_img2img(
        self,
        image,
        timestep,
        batch_size,
        num_images_per_prompt,
        dtype,
        device,
        generator=None,
    ):
        # if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
        #     raise ValueError(
        #         f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
        #     )

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            init_latents = image

        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            elif isinstance(generator, list):
                init_latents = [
                    self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i])
                    for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = self.vae.encode(image).latent_dist.sample(generator)

            init_latents = self.vae.config.scaling_factor * init_latents

        if (
            batch_size > init_latents.shape[0]
            and batch_size % init_latents.shape[0] == 0
        ):
            # expand init_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            # deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat(
                [init_latents] * additional_image_per_prompt, dim=0
            )
        elif (
            batch_size > init_latents.shape[0]
            and batch_size % init_latents.shape[0] != 0
        ):
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # get latents
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
        latents = init_latents

        return latents

    def get_text_cond_grad(
        self,
        prompt=None,
        height=None,
        width=None,
        num_inference_steps=50,
        guidance_scale=7.5,
        negative_prompt=None,
        num_images_per_prompt=1,
        eta=0.0,
        generator=None,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        target_steps=[0],
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        all_token_grads = []
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                if i in target_steps:
                    single_prompt_embeds = prompt_embeds[[0], :, :].clone().detach()
                    single_prompt_embeds.requires_grad = True
                    dummy_prompt_embeds = prompt_embeds[[-1], :, :].clone()

                    input_prompt_embeds = torch.cat(
                        [
                            dummy_prompt_embeds.repeat(num_images_per_prompt, 1, 1),
                            single_prompt_embeds.repeat(num_images_per_prompt, 1, 1),
                        ]
                    )

                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=input_prompt_embeds,
                        cross_attention_kwargs=None,
                        return_dict=False,
                    )[0]

                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred_text = noise_pred_text - noise_pred_uncond
                    noise_pred_text_norm = torch.norm(noise_pred_text, p=2).mean()
                    loss = noise_pred_text_norm

                    (token_grads,) = torch.autograd.grad(loss, [prompt_embeds])
                    token_grads = token_grads.norm(p=2, dim=-1).mean(dim=0).detach()
                    all_token_grads.append(token_grads)

                    with torch.no_grad():
                        noise_pred = (
                            noise_pred_uncond + guidance_scale * noise_pred_text
                        )
                        latents = self.scheduler.step(
                            noise_pred,
                            t,
                            latents,
                            **extra_step_kwargs,
                            return_dict=False,
                        )[0]

                    if i == max(target_steps):
                        torch.cuda.empty_cache()
                        return torch.mean(torch.stack(all_token_grads), dim=0)
                else:
                    with torch.no_grad():
                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=None,
                            return_dict=False,
                        )[0]

                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred_text = noise_pred_text - noise_pred_uncond

                        noise_pred = (
                            noise_pred_uncond + guidance_scale * noise_pred_text
                        )

                        # compute the previous noisy sample x_t -> x_t-1
                        latents = self.scheduler.step(
                            noise_pred,
                            t,
                            latents,
                            **extra_step_kwargs,
                            return_dict=False,
                        )[0]

                progress_bar.update()

    def aug_prompt(
        self,
        prompt=None,
        height=None,
        width=None,
        num_inference_steps=50,
        guidance_scale=7.5,
        negative_prompt=None,
        num_images_per_prompt=1,
        eta=0.0,
        generator=None,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        target_steps=[0],
        lr=0.1,
        optim_iters=10,
        target_loss=None,
        print_optim=False,
        optim_epsilon=None,
        alpha=0.5,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                if i in target_steps:
                    single_prompt_embeds = prompt_embeds[[-1], :, :].clone().detach()
                    if print_optim is True or optim_epsilon is not None:
                        init_embeds = single_prompt_embeds.clone()
                    single_prompt_embeds.requires_grad = True
                    dummy_prompt_embeds = prompt_embeds[[0], :, :].clone()

                    # optimizer
                    optimizer = torch.optim.AdamW([single_prompt_embeds], lr=lr)

                    prompt_tokens = self.tokenizer.encode(prompt)
                    prompt_tokens = prompt_tokens[1:-1]
                    prompt_tokens = prompt_tokens[:75]

                    curr_learnabel_mask = list(set(range(77)) - set([0]))

                    for j in range(optim_iters):
                        if print_optim is True or optim_epsilon is not None:
                            with torch.no_grad():
                                tmp_init_embeds = init_embeds[:, curr_learnabel_mask]
                                tmp_init_embeds = tmp_init_embeds.reshape(
                                    -1, tmp_init_embeds.shape[-1]
                                )
                                tmp_single_prompt_embeds = single_prompt_embeds[
                                    :, curr_learnabel_mask
                                ]
                                tmp_single_prompt_embeds = (
                                    tmp_single_prompt_embeds.reshape(
                                        -1, tmp_single_prompt_embeds.shape[-1]
                                    )
                                )

                                l_inf = torch.norm(
                                    tmp_init_embeds - tmp_single_prompt_embeds,
                                    p=float("inf"),
                                    dim=-1,
                                ).mean()
                                l_2 = torch.norm(
                                    tmp_init_embeds - tmp_single_prompt_embeds,
                                    p=2,
                                    dim=-1,
                                ).mean()

                        input_prompt_embeds = torch.cat(
                            [
                                dummy_prompt_embeds.repeat(num_images_per_prompt, 1, 1),
                                single_prompt_embeds.repeat(
                                    num_images_per_prompt, 1, 1
                                ),
                            ]
                        )

                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=input_prompt_embeds,
                            cross_attention_kwargs=None,
                            return_dict=False,
                        )[0]

                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred_text = noise_pred_text - noise_pred_uncond

                        noise_pred_text_norm = torch.norm(noise_pred_text, p=2).mean()
                        loss = noise_pred_text_norm
                        loss_item = loss.item()

                        if optim_epsilon is not None and l_2 > optim_epsilon:
                            tmp_init_embeds = init_embeds[:, curr_learnabel_mask]
                            tmp_init_embeds = tmp_init_embeds.reshape(
                                -1, tmp_init_embeds.shape[-1]
                            )
                            tmp_single_prompt_embeds = single_prompt_embeds[
                                :, curr_learnabel_mask
                            ]
                            tmp_single_prompt_embeds = tmp_single_prompt_embeds.reshape(
                                -1, tmp_single_prompt_embeds.shape[-1]
                            )

                            loss_l2 = torch.norm(
                                tmp_init_embeds - tmp_single_prompt_embeds, p=2, dim=-1
                            ).mean()

                            loss = alpha * loss + (1 - alpha) * loss_l2

                        if target_loss is not None:
                            if loss_item <= target_loss:
                                if print_optim is True:
                                    print(f"step: {j}, curr loss: {loss_item}")
                                break

                        (single_prompt_embeds.grad,) = torch.autograd.grad(
                            loss, [single_prompt_embeds]
                        )
                        single_prompt_embeds.grad[:, [0]] = (
                            single_prompt_embeds.grad[:, [0]] * 0
                        )

                        optimizer.step()
                        optimizer.zero_grad()

                        if print_optim is True:
                            print(f"step: {j}, curr loss: {loss_item}")

                    single_prompt_embeds = single_prompt_embeds.detach()
                    single_prompt_embeds.requires_grad = False
                    torch.cuda.empty_cache()
                    return single_prompt_embeds

                    with torch.no_grad():
                        noise_pred = (
                            noise_pred_uncond + guidance_scale * noise_pred_text
                        )
                        latents = self.scheduler.step(
                            noise_pred,
                            t,
                            latents,
                            **extra_step_kwargs,
                            return_dict=False,
                        )[0]
                else:
                    with torch.no_grad():
                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=None,
                            return_dict=False,
                        )[0]

                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred_text = noise_pred_text - noise_pred_uncond

                        noise_pred = (
                            noise_pred_uncond + guidance_scale * noise_pred_text
                        )

                        # compute the previous noisy sample x_t -> x_t-1
                        latents = self.scheduler.step(
                            noise_pred,
                            t,
                            latents,
                            **extra_step_kwargs,
                            return_dict=False,
                        )[0]

                progress_bar.update()
