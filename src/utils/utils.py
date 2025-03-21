from ast import literal_eval
from dotenv import load_dotenv, dotenv_values


def load_custom_config():

    custom_config = {
            **dotenv_values("./envs/.env.loss"),
            **dotenv_values("./envs/.env"),
            **dotenv_values("./envs/.env.training"),
            }
    custom_config["FORGET_CLASS"] = literal_eval(custom_config["FORGET_CLASS"])
    return custom_config