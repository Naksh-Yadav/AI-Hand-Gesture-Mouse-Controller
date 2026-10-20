import toml
import os

class Config:
    def __init__(self, config_path="config.toml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            self._config = toml.load(f)

    def get(self, section, key, default=None):
        return self._config.get(section, {}).get(key, default)

    def __getattr__(self, name):
        # Allow accessing sections as attributes
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"Config has no attribute {name}")

# Singleton instance
config = Config()
