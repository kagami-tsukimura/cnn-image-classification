from torch.utils.data import Dataset
from PIL import Image


class CustomDataset(Dataset):
    """Custom dataset class."""

    def __init__(self, img_paths, transform):
        """Initialize the dataset
        Args:
            img_paths: image paths
            transform: data transform
        """
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption).
        Args:
            index: index
        Returns:
            img: image
            img_path: image path
        """

        img_path = self.img_paths[index]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, img_path

    def __len__(self):
        """Returns the total number of image files.
        Returns:
            len: length
        """

        return len(self.img_paths)
