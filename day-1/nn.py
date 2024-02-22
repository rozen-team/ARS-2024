import os
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from torch.autograd import Variable
from torchvision import models

NUMBER = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
ALL_CHAR_SET = NUMBER
ALL_CHAR_SET_LEN = len(ALL_CHAR_SET)
MAX_CAPTCHA = 3


def encode(a):
    onehot = [0] * ALL_CHAR_SET_LEN
    idx = ALL_CHAR_SET.index(a)
    onehot[idx] += 1
    return onehot


class Mydataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.names = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.names[idx])
        image = Image.open(img_name)
        image = image.convert("L")
        str_value = self.names[idx].split("_")[0].replace(".", "")

        label_oh = []
        for i in str_value:
            label_oh += encode(i)

        if self.transform:
            image = self.transform(image)
        image = image + (0.1**0.5) * torch.randn(5, 10, 20)

        return image, np.array(label_oh), str_value


transform = transforms.Compose(
    [
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ]
)

model = models.resnet18(pretrained=False)
model.conv1 = nn.Conv2d(
    1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
)
model.fc = nn.Linear(
    in_features=512, out_features=ALL_CHAR_SET_LEN * MAX_CAPTCHA, bias=True
)

model.load_state_dict(torch.load("day-1/data/model.pth"))
model.eval()


def recognize(img):
    img = transform(img)

    pred = model(img)
    c0 = ALL_CHAR_SET[np.argmax(pred.squeeze().cpu().tolist()[0:ALL_CHAR_SET_LEN])]
    c1 = ALL_CHAR_SET[
        np.argmax(
            pred.squeeze().cpu().tolist()[ALL_CHAR_SET_LEN : ALL_CHAR_SET_LEN * 2]
        )
    ]
    c2 = ALL_CHAR_SET[
        np.argmax(
            pred.squeeze().cpu().tolist()[ALL_CHAR_SET_LEN * 2 : ALL_CHAR_SET_LEN * 3]
        )
    ]

    c = "%s%s%s" % (c0, c1, c2)
    return c
