import modal
import os
import subprocess

def cache_ocr_models():
    """
    This function is run once during the image build process.
    It initializes PaddleOCR, which downloads and caches the required models.
    """
    try:
        from paddleocr import PaddleOCR
        print("Pre-caching PaddleOCR models...")
        # Initializing the class downloads the models to the default cache directory
        PaddleOCR(use_angle_cls=True, lang='en')
        print("PaddleOCR models cached successfully.")
    except ImportError:
        print("PaddleOCR not found in requirements. Could not pre-cache models.")


# 1. Environment Definition
# We use Debian 12 "Bookworm" and add NVIDIA's official repository
# to install the CUDA toolkit, which provides nvcc and a modern GLIBCXX.
image = (
    modal.Image.from_registry(
        "debian:bookworm-slim", add_python="3.10"
    )
    .run_commands(
        # 1. Install prerequisites for adding a new repository
        "apt-get update",
        "apt-get install -y ca-certificates curl gnupg",
        # 2. Add NVIDIA's public key
        "install -m 0755 -d /etc/apt/keyrings",
        "curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb",
        "dpkg -i /tmp/cuda-keyring.deb",
        "rm /tmp/cuda-keyring.deb",
        # 3. Update package list again to include NVIDIA's packages
        "apt-get update",
        # 4. Install the CUDA toolkit from NVIDIA's repository
        "apt-get install -y cuda-toolkit-12-5",
    )
    .apt_install(
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libjpeg-dev",
        "libpng-dev",
        "libtiff-dev",
    )
    .pip_install_from_requirements("requirements.txt")
    .run_function(cache_ocr_models)
)

# 2. Volumes for Persistent Storage and Local Code
# This is for persistent caching of models, etc.
cache_volume = modal.Volume.from_name("flow-grpo-cache", create_if_missing=True)
project_volume = modal.Volume.from_name("flow-grpo-project", create_if_missing=True)

# Upload local project directory to the project volume
with project_volume.batch_upload(force=True) as batch:
    batch.put_directory(".", "/")

# 3. The App
app = modal.App(
    name="flow-grpo",
    image=image,
    # Both volumes are now declared here
    volumes={
        "/root/cache": cache_volume,
        "/root/project": project_volume,
    },
)


@app.function(
    gpu="A100-80GB:8",
    timeout=86400,  # 24 hours
    secrets=[
        modal.Secret.from_name("huggingface-token"),
        modal.Secret.from_name("wandb-token"),
    ],
)
def train():
    import os
    import subprocess

    # Set environment variables for caching
    cache_dir = "/root/cache"
    env = os.environ.copy()
    env["HF_HOME"] = f"{cache_dir}/huggingface"
    env["WANDB_MODE"] = "online"
    env["PYTHONPATH"] = "/root/project"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["OMP_NUM_THREADS"] = "1"
    env["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu/" + os.pathsep + env.get("LD_LIBRARY_PATH", "")

    os.chdir("/root/project")

    # Training command with absolute paths
    cmd = (
        "python -m accelerate.commands.launch "
        "--config_file scripts/accelerate_configs/multi_gpu.yaml "
        "--main_process_port 29501 "
        "scripts/train_sd3.py "
        "--config config/grpo.py:general_ocr_sd3_8gpu"
    )

    # Run the command
    process = subprocess.Popen(cmd, shell=True, env=env)
    process.wait()

    if process.returncode != 0:
        raise Exception(f"Training script failed with exit code {process.returncode}")

    print("Training finished successfully!")
