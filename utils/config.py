import yaml

class Config:
    def __init__(self, config_file):
        with open(config_file, "r") as file:
            self.config = yaml.safe_load(file)

    def __getattr__(self, key):
        return self.config.get(key)

def load_config(config_file):
    return Config(config_file)

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
