# CNN Image Classification

This script performs image classification using Convolutional Neural Networks (CNN). It predicts the images in the `./input` dataset and moves them to output directories classified by their predicted classes.

## Environment Setup

The following libraries are required to run the script:

- tqdm
- torch
- torchvision
- PIL

You can install the required libraries by running the following command:

```bash
pip install -r ./environ/requirements.txt
```

## Data Preparation

Please place the input images you want to classify in the `./input`.

## Usage

To run the script, use the following command:

```bash:
python cnn_eval.py
```

It will utilize the available `GPU` if present; otherwise, it will use the `CPU`.

## Output

The resulting images, classified into their respective classes, will be saved in the `./output/<class_name>`
