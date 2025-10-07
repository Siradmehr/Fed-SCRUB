
import os
from ..utils.utils import load_config, generate_save_path

EXCLUDE = ["sanity_check", "test"]
EXCLUDE = []

f = open("resume.txt", "w")

def gen_res_path(root_path):
    """Yield (dirpath, config) for folders with .env, .env.training, and both model checkpoints, excluding EXCLUDE."""
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Skip if any exclude string is in the path
        if any(ex in dirpath for ex in EXCLUDE):
            continue

        if len(filenames) == 0:
            continue

        if ".env" in filenames[0] and ".env.training" in filenames[1]:
            if "pretrain" not in dirpath:
                continue
            config = load_config(dirpath)
            config["SAVING_DIR"] = generate_save_path(config)
            save_dir = os.path.join(config["SAVING_DIR"], "models_chkpts")
            filename_best = os.path.join(save_dir, "model_best.pth")
            filename_latest = os.path.join(save_dir, "model_latest.pth")
            print("Loaded config from:", dirpath)
            filename_best = filename_best.replace(r"\\", r"/")
            f.writelines("Loaded config from: " + dirpath + ":\n")
            f.writelines(filename_latest)
            f.writelines("\n\n\n\n")


            # print(dirpath)
            print(filename_best)
            # print(filename_latest)
            # if os.path.isfile(filename_best) and os.path.isfile(filename_latest):
            #     print("Loaded config from:", dirpath)
            #     print(filename_best)
            #     print(filename_latest)
            # else:
            #     print("Failed to find model checkpoints in:", dirpath)


# gen_res_path("./envs/ICLR")
# call gen_res_path sys.arg[1]
import sys
path = sys.argv[1]
gen_res_path(path)