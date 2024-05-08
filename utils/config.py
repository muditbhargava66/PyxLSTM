import yaml

class Config:
    def __init__(self, config_file):
        with open(config_file, "r") as file:
            self.config = yaml.safe_load(file)

    def __getattr__(self, key):
        return self.config.get(key)

def load_config(config_file):
    return Config(config_file)