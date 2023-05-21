import torch
import torchvision.models as models
import torch.nn as nn


def load_models(num_classes):
    """
    Load a pre-trained ResNet-18 model.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        model (torch.nn.Module): Loaded ResNet-18 model.
    """
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


def modify_model_weights(resnet, labels):
    """
    Modify the weights of the ResNet-18 model for 2-class classification.

    Args:
        model (torch.nn.Module): ResNet-18 model.
        labels (list): List of original labels.

    Raises:
        ValueError: If the specified labels are not found in the model.

    """
    try:
        # ILSVRC2012: hummingbird (321) and king penguin (346)
        resnet.fc.weight.data[0] = resnet.fc.weight.data[labels.index(321)]
        resnet.fc.weight.data[1] = resnet.fc.weight.data[labels.index(346)]
        resnet.fc.bias.data[0] = resnet.fc.bias.data[labels.index(321)]
        resnet.fc.bias.data[1] = resnet.fc.bias.data[labels.index(346)]
    except ValueError:
        raise ValueError("Specified labels are not found in the model.")


def save_model(model, save_path):
    """
    Save the model to a file.

    Args:
        model (torch.nn.Module): Model to be saved.
        save_path (str): File path to save the model.
    """
    torch.save(model.state_dict(), save_path)


def main():
    # Create and modify the model
    num_classes = 2
    model = load_models(num_classes)
    modify_model_weights(model, [321, 346])

    # Save the model
    save_path = "./weights/resnet_2cls.pt"
    save_model(model, save_path)
    print(model.fc)


if __name__ == "__main__":
    main()
