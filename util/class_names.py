import json
from pathlib import Path
from torchvision.datasets.utils import download_url


def get_classes(CLASS_JSON):
    """Get class names.
    Args:
        CLASS_JSON: class json file
    Returns:
        class_names: class names
    """

    json_path = f"data/{CLASS_JSON}"
    if not Path(json_path).exists():
        # If there is no file, download it.
        download_url("https://git.io/JebAs", "data", CLASS_JSON)

    # Read the class list.
    with open(json_path) as f:
        data = json.load(f)
        class_names = [x["ja"] for x in data]

    return class_names
