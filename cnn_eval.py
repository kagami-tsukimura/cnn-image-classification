import json
import subprocess
from glob import glob
from pathlib import Path
from tqdm import tqdm

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.utils import download_url

from util.custom_dataset import CustomDataset


def prepare_data():
    """Prepare data.
    Returns:
        test_transform: test data transform
    """

    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # The values calculated from the ImageNet dataset.
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return test_transform


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


def mkdir(OUTPUT, dir):
    """Create directory.
    Args:
        dir: directory path
    """

    cmd = f"mkdir -p {OUTPUT}/{dir}"
    subprocess.call(cmd.split())


def mv_file(img, OUTPUT, dir):
    """Move file.
    Args:
        img: image path
        dir: directory path
    """

    cmd = f"mv {img} {OUTPUT}/{dir}"
    subprocess.call(cmd.split())


def main():
    """Main function."""

    CLASS_JSON = "imagenet_classes.json"
    INPUT = "input"
    OUTPUT = "output"
    IMG = ".[jp][pn]g"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.resnet18(pretrained=True).to(device)

    test_transform = prepare_data()
    imgs = glob(f"./{INPUT}/*{IMG}")
    dataset = CustomDataset(imgs, test_transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    class_names = get_classes(CLASS_JSON)

    print(f"Input images: {len(imgs)}")
    print("Start evaluation...")
    for images, img_paths in tqdm(dataloader):
        images = images.to(device)
        model.eval()
        with torch.no_grad():
            output = model(images)

        _, batch_indices = output.sort(dim=1, descending=True)

        for indices, img_path in zip(batch_indices, img_paths):
            dir = class_names[indices[0]]
            mkdir(OUTPUT, dir)
            mv_file(img_path, OUTPUT, dir)
    print("Finish evaluation!")


if __name__ == "__main__":
    main()
