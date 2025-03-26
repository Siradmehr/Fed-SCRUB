from ast import literal_eval
from dotenv import load_dotenv, dotenv_values
from .models import get_model
import torch
import os

import torch
from src.dataloaders.client_dataloader import load_datasets_with_forgetting
# Load configuration
from subprocess import call


def get_gpu():
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    custom_config = load_custom_config()
    print(custom_config)
    device = torch.device(custom_config["DEVICE"] if torch.cuda.is_available() else "cpu")
    print(device)
    torch.cuda.set_device(device)
    call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    print('Active CUDA Device: GPU', torch.cuda.current_device())
    print ('Available devices ', torch.cuda.device_count())
    print ('Current cuda device ', torch.cuda.current_device())
    
def load_custom_config():

    custom_config = {
            **dotenv_values("./envs/.env.loss"),
            **dotenv_values("./envs/.env"),
            **dotenv_values("./envs/.env.training"),
            }
    print(custom_config)
    custom_config["FORGET_CLASS"] = literal_eval(custom_config["FORGET_CLASS"])
    return custom_config


# def setup():
#     custom_config = load_custom_config()
#     config
#     saving_directory = f"./checkpoints/{}"
#     return custom_config


def load_initial_model(custom_config):
    # Configure logging
    model = get_model(custom_config)

    if len(custom_config.get("RESUME", "")) > 0 and custom_config["RESUME"] != "None" and custom_config[
        "RESUME"] != None:
        resume_path = custom_config["RESUME"]
        print(f"Loading model parameters from checkpoint: {resume_path}")

        try:
            if os.path.isfile(resume_path):
                checkpoint = torch.load(resume_path, map_location=torch.device('auto'))

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

                print(f"Checkpoint loaded from {resume_path}")
            else:
                print(f"No checkpoint found at {resume_path}")
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
    else:
        print("Using freshly initialized model (no checkpoint loaded)")

    return model


def save_model(model, custom_config, model_name, epoch=None, optimizer=None, scheduler=None, loss=None, is_best=False):
    """
    Save model checkpoint to the specified path.

    Args:
        model: The model to save
        custom_config: Configuration dictionary containing save path information
        epoch: Current training epoch (optional)
        optimizer: Optimizer state (optional)
        scheduler: Learning rate scheduler state (optional)
        loss: Current loss value (optional)
        is_best: Whether this is the best model so far (optional)
    """
    # Get save directory from config, or use default
    save_dir = custom_config.get("CHECKPOINT_DIR", "./checkpoints")

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Determine filename
    if is_best:
        filename = os.path.join(save_dir, "model_best.pth")
    else:
        if epoch is not None:
            filename = os.path.join(save_dir, f"{model_name}_epoch_{epoch}.pth")
        else:
            filename = os.path.join(save_dir, f"{model_name}_latest.pth")

    # Prepare checkpoint dictionary
    checkpoint = {
        "state_dict": model.state_dict(),
        "config": custom_config
    }

    # Add optional items if provided
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    if loss is not None:
        checkpoint["loss"] = loss

    # Save the checkpoint
    torch.save(checkpoint, filename)
    print(f"Model saved to {filename}")

    return filename