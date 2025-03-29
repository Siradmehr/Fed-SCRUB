import json
from ast import literal_eval
from dotenv import dotenv_values
from .models import get_model
import os
import json
import torch
import numpy as np
import random
# Load configuration
from subprocess import call
import sys

def get_gpu():
    custom_config = load_custom_config()
    print(custom_config)
    device = torch.device(custom_config["DEVICE"] if torch.cuda.is_available() else "cpu")
    print(device)
    torch.cuda.set_device(device)
def load_custom_config_from_files():

    custom_config = {
            **dotenv_values("./envs/.env"),
            **dotenv_values("./envs/.env.training"),
            }
    custom_config["FORGET_CLASS"] = literal_eval(custom_config["FORGET_CLASS"])
    return custom_config

INT_KEYS = ["RETRAIN_BATCH", "FORGET_BATCH", "VAL_BATCH", "TEST_BATCH", "NUM_CLASSES"
            ,"LOCAL_EPOCHS", "MIN_EPOCHS", "MAX_EPOCHS", "LAST_MAX_STEPS",]
def load_custom_config(path: str = "./envs"):
    print("PATH", path)
    custom_config = {
            **dotenv_values(f"{path}/.env"),
            **dotenv_values(f"{path}/.env.training"),
            }
    custom_config["FORGET_CLASS"] = literal_eval(custom_config["FORGET_CLASS"])
    for key in INT_KEYS:
        custom_config[key] = int(custom_config[key])
    if len(custom_config["CLIENT_ID_TO_FORGET"]) > 0:
        custom_config["CLIENT_ID_TO_FORGET"] = [int(i) for i in str(custom_config["CLIENT_ID_TO_FORGET"]).split(",")]
    if len(custom_config["LR_ROUND"]) > 0:
        custom_config["LR_ROUND"] = [int(i) for i in str(custom_config["LR_ROUND"]).split(",")]

    return custom_config

def manual_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup(path: str = "./envs"):
    custom_config = load_custom_config(path)
    saving_directory = f"./checkpoints/{custom_config["CONFIG_ID"]}/{custom_config['MODEL']}/{custom_config["DATASET"]}/{custom_config['CONFIG_NUMBER']}_{custom_config['SEED']}"
    os.makedirs(saving_directory, exist_ok=True)
    custom_config["SAVING_DIR"] = saving_directory
    with open(f"{custom_config['SAVING_DIR']}/custom_config.json", "w") as f:
        f.write(json.dumps(custom_config))

    custom_config["LOADED_MODEL"] = load_initial_model(custom_config['MODEL'], custom_config["RESUME"])
    # TODO LOADING forget_index for the future, retrain_index, valindex, testindex
    return custom_config


def load_initial_model(model_name, path):
    # Configure logging
    model = get_model(model_name)
    print("model initiated")

    if len(path) > 0 and path != "None" and path != None:
        resume_path = path
        print(f"Loading model parameters from checkpoint: {resume_path}")

        try:
            if os.path.isfile(resume_path):
                checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))

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
    save_dir = os.path.join(custom_config["SAVING_DIR"], f"models_chkpts")
    os.makedirs(save_dir, exist_ok=True)
     

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