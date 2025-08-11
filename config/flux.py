import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def get_config():
    config = base.get_config()
    gpu_number = 1

    # -- Pretrained Model --
    config.pretrained.model = "black-forest-labs/FLUX.1-Kontext-dev"

    config.dataset = "data/cleaned/results20k_updated2.csv"

    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt = 8
    config.sample.num_batches_per_epoch = int(8/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 8
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 10
    config.sample.strength = 0.85
    config.sample.global_std = True
    config.sample.mu = 4.0

    # -- Training --
    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.beta = 0.04  # Mirroring the SD3 OCR task
    config.train.ema = True
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99

    # -- Prompting --
    config.prompt_fn = "general_ocr"

    # -- Rewards --
    config.reward_fn = {"ocr": 1.0}

    # -- Logging and Saving --
    config.save_dir = 'logs/ocr/flux'
    config.run_name = "flux_grpo"
    config.save_freq = 2
    config.eval_freq = 10

    return config