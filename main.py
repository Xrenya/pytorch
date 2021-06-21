import os

import cv2
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset

from torchvision import datasets
import torchvision.models as models
from torchvision.transforms import ToTensor

from pytorch_trainer.trainer.trainer import Trainer
import logging

from omegaconf import DictConfig, OmegaConf
import hydra

logging.basicConfig()
logging.root.setLevel(logging.NOTSET)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def run(args):
    print(args.optim)
    logger.info("For logs, checkpoints and samples check")
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }

    class CustomImageDataset(Dataset):
        def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
            self.img_labels = pd.read_csv(annotations_file)
            self.img_dir = img_dir
            self.transform = transform
            self.target_transform = target_transform

        def __len__(self):
            return len(self.img_labels)

        def __getitem__(self, idx):
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
            image = read_image(img_path)
            label = self.img_labels.iloc[idx, 1]
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            return image, label


    train_dataloader = DataLoader(training_data,
                                  batch_size=args.batch_size,
                                  shuffle=True)
    test_dataloader = DataLoader(test_data,
                                 batch_size=args.batch_size,
                                 shuffle=True)

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 10)
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    learning_rate = 1e-3
    epochs = 5
    loss_fn = nn.CrossEntropyLoss()
    model = NeuralNetwork()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)


    def metrics(output, labels):
        correct = (torch.argmax(output, dim=1) == labels).type(torch.float).sum().item()
        return correct

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = Trainer(model, loss_fn, optimizer, metrics, device, train_dataloader, test_dataloader)

    output = trainer.train(args.epochs)
    print(output)

@hydra.main(config_path="conf", config_name='config.yaml')
def main(args):
    try:
        ran(args)
    except Exception:
        logger.exception("Some error happened")
        # Hydra intercepts exit code, fixed in beta but I could not get the beta to work
        os._exit(1)

if __name__ == "__main__":
    main()