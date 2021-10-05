import yaml
from box import Box

with open("configs/configuration.yml", "r") as ymlfile:
  cfg = Box(yaml.safe_load(ymlfile))
