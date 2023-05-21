from glob import glob
from tqdm import tqdm

import torch
import torchvision
from torch.utils.data import DataLoader

from util.custom_dataset import CustomDataset
from util.prepare_data import prepare_data
from util.class_names import get_classes
from util.file_controls import mkdir, mv_file


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
