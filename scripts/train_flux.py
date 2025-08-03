import csv
import os
import io
import random
import time
import json
import contextlib
from collections import defaultdict
from concurrent import futures
import tempfile
from absl import app, flags
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration
from diffusers import FluxPipeline, FluxTransformer2DModel
from diffusers.utils.torch_utils import is_compiled_module
from ml_collections import config_flags
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from PIL import Image
import requests
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import wandb

import flow_grpo.rewards
from flow_grpo.ema import EMAModuleWrapper
from flow_grpo.stat_tracking import PerPromptStatTracker

# --- Configuration Flags ---
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/flux.py", "Training configuration.")

logger = get_logger(__name__)

# --- Constants ---
URL_COLUMN = "product_url"
DESCRIPTION_COLUMN = "product_showcase_description"
OCR_COLUMN = "paddle_ocr_detection"
ID_COLUMN = "id"
IMAGE_CACHE_DIR = "data/images"

# --- Dataset ---
class ImagePromptDataset(Dataset):
    def __init__(self, csv_path, tokenizer, split='train', eval_count=1000):
        self.csv_path = csv_path
        self.tokenizer = tokenizer
        self.split = split
        self.eval_count = eval_count
        self.image_cache_dir = IMAGE_CACHE_DIR
        os.makedirs(self.image_cache_dir, exist_ok=True)

        try:
            with open(self.csv_path, 'r', newline='', encoding='utf-8') as infile:
                reader = csv.DictReader(infile)
                all_rows = list(reader)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset CSV not found at: {self.csv_path}")

        if split == 'train':
            self.rows = all_rows[:-self.eval_count]
        else:
            self.rows = all_rows[-self.eval_count:]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image_url = row.get(URL_COLUMN)
        image_id = row.get(ID_COLUMN)

        image = self._get_image_from_cache(image_url, image_id)
        prompt = self._construct_prompt(row)
        prompt_ids = self.tokenizer(prompt, padding="max_length", max_length=256, truncation=True, return_tensors="pt").input_ids

        return {"prompt": prompt, "image": image, "prompt_ids": prompt_ids.squeeze(), "metadata": {"id": image_id}}

    def _get_image_from_cache(self, image_url, image_id):
        image_filename = f"{image_id}.png"
        cached_image_path = os.path.join(self.image_cache_dir, image_filename)

        if os.path.exists(cached_image_path):
            try:
                return Image.open(cached_image_path).convert("RGB")
            except IOError:
                pass

        try:
            response = requests.get(image_url, stream=True, timeout=10)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            image.save(cached_image_path)
            return image
        except (requests.RequestException, IOError, Image.UnidentifiedImageError):
            return None

    def _construct_prompt(self, row):
        description = row.get(DESCRIPTION_COLUMN, "")
        product_text = row.get(OCR_COLUMN, "")
        start_prompt = f"Follow this specific direction for the composition: {description}. " if description else ""
        basic_prompt = "Create a photorealistic scene or background that best suits and complements the product..."
        product_text_prompt = f'Put the following text clearly and readable on the product: "{product_text}". DO NOT CHANGE THE TEXT.' if product_text else ""
        return f"{start_prompt}{basic_prompt} {product_text_prompt}".strip()

    @staticmethod
    def collate_fn(examples):
        examples = [e for e in examples if e["image"] is not None]
        if not examples:
            return None, None, None, None
        prompts = [e["prompt"] for e in examples]
        images = [e["image"] for e in examples]
        prompt_ids = torch.stack([e["prompt_ids"] for e in examples])
        metadatas = [e["metadata"] for e in examples]
        return prompts, images, prompt_ids, metadatas

# --- Log Prob & Helper Functions ---
def flux_sde_step_with_logprob(scheduler, model_output, timestep, sample, prev_sample):
    sigma_t_index = (scheduler.timesteps == timestep).nonzero().item()
    sigma_t = scheduler.sigmas[sigma_t_index]
    sigma_s_index = sigma_t_index + 1
    sigma_s = scheduler.sigmas[sigma_s_index] if sigma_s_index < len(scheduler.sigmas) else 0.0

    denoised = sample - model_output * sigma_t
    dt = sigma_s - sigma_t
    prev_sample_mean = sample + (denoised - sample) / sigma_t * dt
    std_dev_t = torch.sqrt((sigma_s**2 * (sigma_t**2 - sigma_s**2)) / sigma_t**2)
    
    variance = std_dev_t**2
    log_prob = -0.5 * (((prev_sample - prev_sample_mean) ** 2 / variance).sum(dim=(1, 2, 3)) + 
                      (torch.log(2 * torch.pi * variance)).sum(dim=(1, 2, 3)))
    return prev_sample_mean, log_prob, std_dev_t

def flux_pipeline_with_logprob(pipeline, prompt, image, strength, num_inference_steps, generator, guidance_scale, neg_prompt_embeds, neg_pooled_prompt_embeds):
    prompt_embeds, pooled_prompt_embeds = pipeline.encode_prompt(prompt)
    latents = pipeline.prepare_latents(image=image, strength=strength, batch_size=len(prompt), dtype=pipeline.transformer.dtype, device=pipeline.device, generator=generator)
    
    pipeline.scheduler.set_timesteps(num_inference_steps, device=pipeline.device)
    timesteps = pipeline.scheduler.timesteps

    all_latents = [latents.clone()]
    all_log_probs = []

    for t in timesteps:
        latent_model_input = torch.cat([latents] * 2)
        encoder_hidden_states = torch.cat([neg_prompt_embeds, prompt_embeds])
        pooled_projections = torch.cat([neg_pooled_prompt_embeds, pooled_prompt_embeds])

        model_pred_chunks = pipeline.transformer(latent_model_input, encoder_hidden_states=encoder_hidden_states, pooled_projections=pooled_projections, timestep=t).sample
        noise_pred_uncond, noise_pred_text = model_pred_chunks.chunk(2)
        
        # Correctly calculate the log_prob using the UN-GUIDED model prediction
        _, log_prob, _ = flux_sde_step_with_logprob(pipeline.scheduler, noise_pred_text, t, latents, pipeline.scheduler.step(noise_pred_text, t, latents).prev_sample)
        all_log_probs.append(log_prob)

        # Perform guidance for the actual step
        guided_model_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = pipeline.scheduler.step(guided_model_pred, t, latents, generator=generator).prev_sample
        all_latents.append(latents.clone())

    images = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor, return_dict=False)[0]
    images = pipeline.image_processor.postprocess(images, output_type="pt")

    return images, all_latents, all_log_probs

def compute_flux_embeddings(prompt, pipeline, device):
    prompt_embeds, pooled_prompt_embeds = pipeline.encode_prompt(prompt)
    return prompt_embeds.to(device), pooled_prompt_embeds.to(device)

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    return model._orig_mod if is_compiled_module(model) else model

def save_ckpt(save_dir, transformer, global_step, accelerator, ema, config):
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    os.makedirs(save_root, exist_ok=True)
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(list(filter(lambda p: p.requires_grad, transformer.parameters())), store_temp=True)
        unwrap_model(transformer, accelerator).save_pretrained(save_root)
        if config.train.ema:
            ema.copy_temp_to(list(filter(lambda p: p.requires_grad, transformer.parameters())))

def eval_model(pipeline, tokenizer, dataloader, config, accelerator, reward_fn, global_step, ema):
    if config.train.ema:
        ema.copy_ema_to(list(filter(lambda p: p.requires_grad, pipeline.transformer.parameters())), store_temp=True)

    all_rewards = defaultdict(list)
    all_prompts = []
    all_generated_images = []

    neg_prompt_embeds, neg_pooled_prompt_embeds = compute_flux_embeddings([""], pipeline, accelerator.device)
    neg_prompt_embeds = neg_prompt_embeds.repeat(config.sample.batch_size, 1, 1)
    neg_pooled_prompt_embeds = neg_pooled_prompt_embeds.repeat(config.sample.batch_size, 1)

    for batch in tqdm(dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process):
        if batch[0] is None: continue
        prompts, images, _, metadatas = batch
        with torch.no_grad():
            generated_images, _, _ = flux_pipeline_with_logprob(
                pipeline, prompt=prompts, image=images, strength=config.sample.strength, 
                num_inference_steps=config.sample.eval_num_steps, generator=torch.Generator(device=accelerator.device).manual_seed(config.seed),
                guidance_scale=config.sample.guidance_scale, neg_prompt_embeds=neg_prompt_embeds[:len(prompts)], neg_pooled_prompt_embeds=neg_pooled_prompt_embeds[:len(prompts)]
            )
        rewards, _ = reward_fn(generated_images, prompts, metadatas)
        
        gathered_images = accelerator.gather(generated_images)
        gathered_prompts = accelerator.gather(torch.tensor(tokenizer(prompts, padding="max_length", truncation=True, max_length=256).input_ids, device=accelerator.device))
        all_generated_images.append(gathered_images.cpu())
        all_prompts.append(gathered_prompts.cpu())

        for key, value in rewards.items():
            all_rewards[key].append(accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy())

    if accelerator.is_main_process:
        all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
        all_prompts = tokenizer.batch_decode(torch.cat(all_prompts), skip_special_tokens=True)
        all_generated_images = torch.cat(all_generated_images)

        wandb.log({f"eval_reward_{key}": np.mean(value) for key, value in all_rewards.items()}, step=global_step)

        with tempfile.TemporaryDirectory() as tmpdir:
            for i, (img, prompt) in enumerate(zip(all_generated_images, all_prompts)):
                if i >= 16: break
                pil = Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                pil.save(os.path.join(tmpdir, f"{i}.jpg"))
            wandb.log({"eval_images": [wandb.Image(os.path.join(tmpdir, f"{i}.jpg"), caption=all_prompts[i]) for i in range(min(16, len(all_generated_images)))]}, step=global_step)

    if config.train.ema:
        ema.copy_temp_to(list(filter(lambda p: p.requires_grad, pipeline.transformer.parameters())))

def compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, neg_embeds, neg_pooled_embeds, config):
    latent_model_input = torch.cat([sample["latents"][:, j]] * 2)
    encoder_hidden_states = torch.cat([neg_embeds, embeds])
    pooled_projections = torch.cat([neg_pooled_embeds, pooled_embeds])

    model_pred_chunks = transformer(latent_model_input, encoder_hidden_states=encoder_hidden_states, pooled_projections=pooled_projections, timestep=sample["timesteps"][:, j]).sample
    noise_pred_uncond, noise_pred_text = model_pred_chunks.chunk(2)
    
    # Re-calculate the guided prediction to match the sampling process
    guided_model_pred = noise_pred_uncond + config.sample.guidance_scale * (noise_pred_text - noise_pred_uncond)

    # The log prob should be based on the guided prediction if that's what the action (next_latent) was based on
    prev_sample_mean, log_prob, std_dev_t = flux_sde_step_with_logprob(pipeline.scheduler, guided_model_pred, sample["timesteps"][:, j], sample["latents"][:, j], sample["next_latents"][:, j])
    return prev_sample_mean, log_prob, std_dev_t

# --- Main Training Logic ---
def main(_):
    config = FLAGS.config
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)
    accelerator = Accelerator(log_with="wandb", mixed_precision=config.mixed_precision, gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps)

    if accelerator.is_main_process:
        wandb.init(project="flow_grpo_flux", name=config.run_name, config=config.to_dict())

    set_seed(config.seed, device_specific=True)

    # --- Model & Tokenizer Loading ---
    device = accelerator.device
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained.model, subfolder="tokenizer")
    transformer = FluxTransformer2DModel.from_pretrained(config.pretrained.model, torch_dtype=torch.bfloat16)
    pipeline = FluxPipeline.from_pretrained(config.pretrained.model, tokenizer=tokenizer, transformer=transformer, torch_dtype=torch.bfloat16)
    pipeline.to(device)

    if config.use_lora:
        lora_config = LoraConfig(r=32, lora_alpha=64, target_modules=["attn.to_q", "attn.to_k", "attn.to_v"])
        transformer = get_peft_model(transformer, lora_config)

    transformer_trainable_params = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    optimizer = torch.optim.AdamW(transformer_trainable_params, lr=config.train.learning_rate)
    ema = EMAModuleWrapper(transformer_trainable_params, decay=0.9)

    # --- Dataset & DataLoader ---
    train_dataset = ImagePromptDataset(config.dataset_path, tokenizer, split='train')
    train_sampler = DistributedSampler(train_dataset, num_replicas=accelerator.num_processes, rank=accelerator.process_index, shuffle=True, seed=config.seed)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=config.sample.batch_size, collate_fn=ImagePromptDataset.collate_fn)
    eval_dataset = ImagePromptDataset(config.dataset_path, tokenizer, split='eval')
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.sample.batch_size, collate_fn=ImagePromptDataset.collate_fn)

    # --- Prepare with Accelerator ---
    transformer, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(transformer, optimizer, train_dataloader, eval_dataloader)

    reward_fn = flow_grpo.rewards.multi_score(device, config.reward_fn)
    eval_reward_fn = flow_grpo.rewards.multi_score(device, config.reward_fn)
    stat_tracker = PerPromptStatTracker(config.sample.global_std)
    executor = futures.ThreadPoolExecutor(max_workers=8)

    neg_prompt_embeds, neg_pooled_prompt_embeds = compute_flux_embeddings([""], accelerator.unwrap_model(pipeline), device)
    sample_neg_prompt_embeds = neg_prompt_embeds.repeat(config.sample.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embeds.repeat(config.sample.batch_size, 1)
    train_neg_prompt_embeds = neg_prompt_embeds.repeat(config.train.batch_size, 1, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embeds.repeat(config.train.batch_size, 1)

    # --- Training Loop ---
    global_step = 0
    train_iter = iter(train_dataloader)

    for epoch in range(1000):
        if epoch > 0 and epoch % config.eval_freq == 0:
            eval_model(accelerator.unwrap_model(pipeline), tokenizer, eval_dataloader, config, accelerator, eval_reward_fn, global_step, ema)
        if epoch > 0 and epoch % config.save_freq == 0:
            save_ckpt(config.logdir, transformer, global_step, accelerator, ema, config)

        samples = []
        for i in tqdm(range(config.sample.num_batches_per_epoch), desc=f"Epoch {epoch}: Sampling"):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                batch = next(train_iter)

            if batch[0] is None: continue
            prompts, images, prompt_ids, metadatas = batch

            prompt_embeds, pooled_prompt_embeds = compute_flux_embeddings(prompts, accelerator.unwrap_model(pipeline), device)

            with torch.no_grad():
                generated_images, latents, log_probs = flux_pipeline_with_logprob(
                    accelerator.unwrap_model(pipeline), prompt=prompts, image=images, strength=config.sample.strength,
                    num_inference_steps=config.sample.num_steps, generator=torch.Generator(device=device).manual_seed(config.seed),
                    guidance_scale=config.sample.guidance_scale, neg_prompt_embeds=sample_neg_prompt_embeds[:len(prompts)], neg_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds[:len(prompts)]
                )
            
            rewards_future = executor.submit(reward_fn, generated_images, prompts, metadatas)
            samples.append({
                "prompt_ids": prompt_ids, "prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds,
                "timesteps": pipeline.scheduler.timesteps.repeat(len(prompts), 1),
                "latents": torch.stack(latents[:-1], dim=1), "next_latents": torch.stack(latents[1:], dim=1),
                "log_probs": torch.stack(log_probs, dim=1), "rewards_future": rewards_future
            })

        for sample in tqdm(samples, desc="Waiting for rewards"):
            rewards, _ = sample["rewards_future"].result()
            sample["rewards"] = {k: torch.as_tensor(v, device=device) for k, v in rewards.items()}
            del sample["rewards_future"]

        samples_collated = {k: torch.cat([s[k] for s in samples]) for k in samples[0] if k not in ["prompts", "rewards"]}
        samples_collated["rewards"] = {k: torch.cat([s["rewards"][k] for s in samples]) for k in samples[0]["rewards"]}
        prompts_collated = [p for s in samples for p in s["prompts"]]

        gathered_rewards = {k: accelerator.gather(v).cpu().numpy() for k, v in samples_collated["rewards"].items()}
        advantages = stat_tracker.update(prompts_collated, gathered_rewards['avg'])
        advantages = torch.as_tensor(advantages, device=device).reshape(accelerator.num_processes, -1)[accelerator.process_index]
        samples_collated["advantages"] = advantages.unsqueeze(1).repeat(1, num_train_timesteps)

        for inner_epoch in range(config.train.num_inner_epochs):
            perm = torch.randperm(samples_collated["latents"].shape[0])
            samples_shuffled = {k: v[perm] for k, v in samples_collated.items() if torch.is_tensor(v)}

            for i in range(0, samples_shuffled["latents"].shape[0], config.train.batch_size):
                batch_slice = slice(i, i + config.train.batch_size)
                info = defaultdict(list)
                for j in range(num_train_timesteps):
                    with accelerator.accumulate(transformer):
                        prev_sample_mean, log_prob, std_dev_t = compute_log_prob(transformer, accelerator.unwrap_model(pipeline), samples_shuffled, j, samples_shuffled["prompt_embeds"][batch_slice], samples_shuffled["pooled_prompt_embeds"][batch_slice], train_neg_prompt_embeds[:len(samples_shuffled["prompt_embeds"][batch_slice])], train_neg_pooled_prompt_embeds[:len(samples_shuffled["prompt_embeds"][batch_slice])], config)
                        
                        with torch.no_grad():
                            with unwrap_model(transformer, accelerator).disable_adapter():
                                prev_sample_mean_ref, _, _ = compute_log_prob(transformer, accelerator.unwrap_model(pipeline), samples_shuffled, j, samples_shuffled["prompt_embeds"][batch_slice], samples_shuffled["pooled_prompt_embeds"][batch_slice], train_neg_prompt_embeds[:len(samples_shuffled["prompt_embeds"][batch_slice])], train_neg_pooled_prompt_embeds[:len(samples_shuffled["prompt_embeds"][batch_slice])], config)

                        advantages_batch = torch.clamp(samples_shuffled["advantages"][batch_slice, j], -config.train.adv_clip_max, config.train.adv_clip_max)
                        ratio = torch.exp(log_prob - samples_shuffled["log_probs"][batch_slice, j])
                        unclipped_loss = -advantages_batch * ratio
                        clipped_loss = -advantages_batch * torch.clamp(ratio, 1.0 - config.train.clip_range, 1.0 + config.train.clip_range)
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        
                        kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean() / (2 * std_dev_t ** 2)
                        loss = policy_loss + config.train.beta * kl_loss
                        
                        info["loss"].append(loss)
                        info["policy_loss"].append(policy_loss)
                        info["kl_loss"].append(kl_loss)
                        info["approx_kl"].append(0.5 * torch.mean((log_prob - samples_shuffled["log_probs"][batch_slice, j]) ** 2))
                        info["clipfrac"].append(torch.mean((torch.abs(ratio - 1.0) > config.train.clip_range).float()))

                        accelerator.backward(loss)
                        optimizer.step()
                        optimizer.zero_grad()

                if accelerator.sync_gradients:
                    global_step += 1
                    if accelerator.is_main_process:
                        log_info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        log_info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        wandb.log(log_info, step=global_step)

if __name__ == "__main__":
    app.run(main)