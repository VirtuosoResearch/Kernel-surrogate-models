import yaml

# ConfigDict is a subclass of dict that allows you to access keys as attributes
# TODO JL 10/26/24 add handling of none etc
class ConfigDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = ConfigDict(value)

    def __getattr__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        if "lr" in key or "decay" in key:
            value = float(value)
        if isinstance(value, dict):
            value = ConfigDict(value, keys_to_float=self.keys_to_float)
        super().__setitem__(key, value)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def copy(self):
        copied_dict = ConfigDict()
        for key, value in self.items():
            if isinstance(value, dict):
                copied_dict[key] = value.copy()
            else:
                copied_dict[key] = value
        return copied_dict

    def override(self, other):
        for key, value in other.items():
            if isinstance(value, dict) and key in self and isinstance(self[key], ConfigDict):
                self[key].override(value)
            else:
                self[key] = value

    # TODO JL 10/25/24 override with argparse

def load_config(task):
    path = f'./src/configs/tasks/{task}.yaml'
    with open(path, "r") as f:
        config = yaml.load(f, yaml.FullLoader)
    return ConfigDict(config)