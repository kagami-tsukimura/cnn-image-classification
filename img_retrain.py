import subprocess
from tqdm import tqdm
from glob import glob

import torchvision.models as models
from PIL import ImageFile
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms


def setting_backborn():
    """Model settings.
    Returns:
        model: model
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True).to(device)
    for param in model.parameters():
        param.requires_grad = True
    # Multi-class classification
    model.fc = nn.Linear(in_features=512, out_features=2).to(device, non_blocking=True)
    device_ids = []
    for i in range(torch.cuda.device_count()):
        device_ids.append(i)
    # Multi GPU
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    return model


def mkdirs(OUTPUT, DIRS):
    """Create directory.
    Args:
        DIRS: directory path
    """

    for dir in DIRS:
        cmd = f"mkdir -p {OUTPUT}/{dir}"
        subprocess.call(cmd.split())


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


def eval_cnn(img, test_transform, model):
    """CNN image classification.
    Args:
        img: image path
        test_transform: test data transform
        model: model
    Returns:
        pred: prediction result
    """

    test_img = Image.open(img).convert("RGB")
    test_img_tensor = test_transform(test_img).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(test_img_tensor)
        pred = torch.argmax(output).item()
    return pred


def judge_pred(pred, DIRS):
    """Judge prediction result.
    Args:
        pred: prediction result
        DIRS: directory path
    Returns:
        dst_dir: destination directory path
    """

    if pred == 0:
        dst_dir = DIRS[0]
    elif pred == 1:
        dst_dir = DIRS[1]
    return dst_dir


def mv_file(img, OUTPUT, dst_dir):
    """Move file.
    Args:
        img: image path
        dst_dir: destination directory path
    """

    cmd = f"mv {img} {OUTPUT}/{dst_dir}"
    subprocess.call(cmd.split())


def main():
    """Main function."""

    MODEL = "./weights/resnet_2cls.pt"
    INPUT = ["input"]
    OUTPUT = "output"
    # ILSVRC2012: hummingbird (321) and King penguin (346)
    LABELS = [
        "0_hummingbird",
        "1_king_penguin",
    ]
    mkdirs(OUTPUT, LABELS)

    for input in INPUT:
        for label in LABELS:
            LABEL_PATH = f"./{input}/{label}"
            model = setting_backborn()
            model.load_state_dict(torch.load(MODEL))
            imgs = glob(f"./{LABEL_PATH}/*.[jp][pn]g")
            print(f"{label} images: {len(imgs)}")

            # Can be read with large images.
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            test_transform = prepare_data()

            for img in tqdm(imgs):
                pred = eval_cnn(img, test_transform, model)
                dst_dir = judge_pred(pred, LABELS)
                mv_file(img, OUTPUT, dst_dir)


if __name__ == "__main__":
    main()
