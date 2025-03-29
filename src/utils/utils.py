import json
import os
import torch
import numpy as np
import random
from ast import literal_eval
from dotenv import dotenv_values
from typing import Dict, List, Optional, Union

from .models import get_model

# Constants
INT_KEYS = [
    "RETRAIN_BATCH", "FORGET_BATCH", "VAL_BATCH", "TEST_BATCH",
    "NUM_CLASSES", "LOCAL_EPOCHS", "MIN_EPOCHS", "MAX_EPOCHS", "LAST_MAX_STEPS"
]


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(config: Dict) -> torch.device:
    """Get the appropriate device based on configuration and availability."""
    device_name = config.get("DEVICE", "cuda:0") if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    print(f"Using device: {device}")
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    return device


def load_config(path: str = "./envs") -> Dict:
    """Load and process configuration from environment files."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration directory not found: {path}")

    # Load configuration files
    env_path = os.path.join(path, ".env")
    training_path = os.path.join(path, ".env.training")

    if not os.path.exists(env_path) or not os.path.exists(training_path):
        raise FileNotFoundError(f"Required configuration files missing in {path}")

    config = {
        **dotenv_values(env_path),
        **dotenv_values(training_path),
    }

    # Process configuration values
    try:
        # Handle forget class
        if "FORGET_CLASS" in config:
            config["FORGET_CLASS"] = literal_eval(config["FORGET_CLASS"])

        # Convert integer keys
        for key in INT_KEYS:
            if key in config:
                config[key] = int(config[key])

        # Process comma-separated integer lists
        for key in ["CLIENT_ID_TO_FORGET", "LR_ROUND"]:
            if key in config and config[key]:
                config[key] = [int(i) for i in str(config[key]).split(",")]
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Error parsing configuration: {e}")

    return config


def setup_experiment(path: str = "./envs") -> Dict:
    """Set up the experiment with configuration, directories, and model."""
    # Load configuration
    config = load_config(path)

    # Create saving directory
    saving_directory = os.path.join(
        "./checkpoints",
        config["CONFIG_ID"],
        config["MODEL"],
        config["DATASET"],
        f"{config['CONFIG_NUMBER']}_{config['SEED']}"
    )
    os.makedirs(saving_directory, exist_ok=True)
    config["SAVING_DIR"] = saving_directory

    # Save configuration
    config_path = os.path.join(saving_directory, "custom_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    # Load initial model
    config["LOADED_MODEL"] = load_model(config["MODEL"], config.get("RESUME", ""))

    return config


def load_model(model_name: str, checkpoint_path: Optional[str] = None) -> torch.nn.Module:
    """Load a model and initialize from checkpoint if provided."""
    model = get_model(model_name)
    print(f"Model '{model_name}' initialized")

    if not checkpoint_path or checkpoint_path in ("None", ""):
        print("Using freshly initialized model (no checkpoint loaded)")
        return model

    if not os.path.isfile(checkpoint_path):
        print(f"Warning: No checkpoint found at {checkpoint_path}")
        return model

    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
            print("Successfully loaded model state_dict")
        elif "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
            print("Successfully loaded model weights")
        else:
            # Try loading directly
            model.load_state_dict(checkpoint)
            print("Successfully loaded model weights directly")

        print(f"Checkpoint loaded from {checkpoint_path}")
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")

    return model


def save_model(
        model: torch.nn.Module,
        config: Dict,
        round: Optional[int] = None,
        is_best: bool = False
) -> str:
    """Save model checkpoint to specified path."""
    # Create save directory
    save_dir = os.path.join(config["SAVING_DIR"], "models_chkpts")
    os.makedirs(save_dir, exist_ok=True)

    # Determine filename
    if is_best:
        filename = os.path.join(save_dir, "model_best.pth")
    elif round is not None:
        filename = os.path.join(save_dir, f"model_round_{round}.pth")
    else:
        filename = os.path.join(save_dir, "model_latest.pth")

    # Prepare and save checkpoint
    checkpoint = {"state_dict": model.state_dict()}
    torch.save(checkpoint, filename)
    print(f"Model saved to {filename}")

    return filename