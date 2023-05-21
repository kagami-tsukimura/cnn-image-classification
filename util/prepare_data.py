from torchvision import transforms


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
