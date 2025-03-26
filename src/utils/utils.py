import json
from ast import literal_eval
from dotenv import load_dotenv, dotenv_values
from .models import get_model
import torch
import os
import json
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
            **dotenv_values("./envs/.env"),
            **dotenv_values("./envs/.env.training"),
            }
    print(custom_config)
    custom_config["FORGET_CLASS"] = literal_eval(custom_config["FORGET_CLASS"])
    return custom_config


def setup():
    custom_config = load_custom_config()
    saving_directory = f"./checkpoints/{custom_config["CONFIG_ID"]}/{custom_config['MODEL_NAME']}/\
    /{custom_config["DATASET"]}/{custom_config['CONFIG_NUMBER']}_{custom_config['SEED']}"
    os.makedirs(saving_directory, exist_ok=True)
    custom_config["SAVING_DIR"] = saving_directory
    with open(f"{custom_config['SAVING_DIR']}/custom_config.json") as f:
        f.write(json.dumps(custom_config))

    custom_config["LOADED_MODEL"] = load_initial_model(custom_config['MODEL_NAME'], custom_config["RESUME"])
    # TODO LOADING forget_index for the future, retrain_index, valindex, testindex
    return custom_config


def load_initial_model(model_name, path):
    # Configure logging
    model = get_model(model_name)

    if len(path) > 0 and path != "None" and path != None:
        resume_path = path
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


def save_model(model, custom_config, round=None, is_best=False):
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
    save_dir = custom_config["SAVING_DIR"]

    # Determine filename
    if is_best:
        filename = os.path.join(save_dir, "model_best.pth")
    else:
        if round is not None:
            filename = os.path.join(save_dir, f"model_round_{round}.pth")
        else:
            filename = os.path.join(save_dir, f"model_latest.pth")

    # Prepare checkpoint dictionary
    checkpoint = {
        "state_dict": model.state_dict(),
    }

    # Save the checkpoint
    torch.save(checkpoint, filename)
    print(f"Model saved to {filename}")

    return filename