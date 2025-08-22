import json
import os

class Logger:
    
    def __init__(self, label, dir):
        self.label = label
        self.path = os.path.join(
            dir,
            f"{label}.json",
        )
        self.data = {}

    def log(self, key, value, iter):
        if key not in self.data:
            self.data[key] = {
                "iter": [],
                "value": [],
            }
        self.data[key]["iter"].append(iter)
        self.data[key]["value"].append(value)
        # print(f"Logging {key}: {value} at iter {iter}")
    
    def reset(self, key):
        self.data[key] = {
            "iter": [],
            "value": [],
        }

    def log_value(self, key, value):
        if key not in self.data:
            self.data[key] = {
                "value": value,
            }
        else:
            self.data[key]["value"] = value
        # print(f"Logging {key}: {value} at iter {iter}")

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f)

    def load(self):
        with open(self.path, "r") as f:
            self.data = json.load(f)
        return self.data

    def get(self, key):
        return self.data[key]
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __setitem__(self, key, value):
        self.data[key] = value
    
    def __delitem__(self, key):
        del self.data[key]
    
    def __contains__(self, key):
        return key in self.data
