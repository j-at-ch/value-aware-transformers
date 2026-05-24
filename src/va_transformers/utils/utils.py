import yaml
from pathlib import Path
from typing import Dict, List, Union


def read_yaml(path: Path) -> Union[Dict, List]:
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config
