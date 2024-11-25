import os
import json
from logger import Logger

class GenericFunction:
    def __init__(self):
        """Initialize and load config.json if present."""
        self.logger = Logger()
        self.config_data = {}
        self.load_config()
        

    def load_config(self):
        """Load the config.json file from the current directory."""
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config_data = json.load(f)
            self.logger.log_info(f"Loaded config.json from: {config_path}")
        else:
            self.logger.log_error("config.json not found in the current directory")
            raise FileNotFoundError("config.json not found in the current directory.")

    def get_value(self, key):
        """
        Retrieve a value from the loaded config.json by key.
        """
        if key in self.config_data:
            return self.config_data[key]
        else:
            raise KeyError(f"Key '{key}' not found in config.json.")

    def read_prompt(self,file_path):
        with open(file_path, 'r') as file:
            return file.read().strip()

