
import ml_collections
import os

def get_config():
    config = ml_collections.ConfigDict()

    # -- Pretrained Model --
    config.pretrained = ml_collections.ConfigDict()
    config.pretrained.model = "black-forest-labs/FLUX.1-Kontext-dev"

    # -- Dataset --
    config.dataset_path = "data/processed/results20k.csv" # Path to the processed CSV

    # -- LoRA --
    config.use_lora = True

    # -- Sampling --
    config.sample = ml_collections.ConfigDict()
    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4
    config.sample.num_steps = 20
    config.sample.eval_num_steps = 20
    config.sample.guidance_scale = 4.5
    config.sample.strength = 0.85 # For image-to-image

    # -- Training --
    config.train = ml_collections.ConfigDict()
    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 1.0
    config.train.beta = 0.04
    config.train.ema = True
    config.train.learning_rate = 1e-5
    config.train.adam_beta1 = 0.9
    config.train.adam_beta2 = 0.999
    config.train.adam_weight_decay = 1e-2
    config.train.adam_epsilon = 1e-8
    config.train.max_grad_norm = 1.0
    config.train.lora_path = ""
    config.train.use_8bit_adam = False
    config.train.clip_range = 0.2
    config.train.adv_clip_max = 5

    # -- Prompting --
    config.prompt_fn = "flux_image_prompt"

    # -- Rewards --
    config.reward_fn = {"ocr": 1.0}
    config.per_prompt_stat_tracking = True

    # -- Logging and Saving --
    config.logdir = "logs"
    config.run_name = "flux_grpo_training"
    config.save_freq = 10
    config.eval_freq = 10
    config.num_checkpoint_limit = 5
    config.mixed_precision = "fp16"
    config.allow_tf32 = True
    config.seed = 42

    return config
