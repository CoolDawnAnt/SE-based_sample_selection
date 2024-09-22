import os
import clip
import torch
from torchvision.datasets import CIFAR10
from torchvision import datasets
import numpy as np
from tqdm import trange
import math
import argparse

import scipy
from scipy import sparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR10,CIFAR100 Training')

######################### Training Setting #########################
parser.add_argument('--store-dir',type=str,default = '../graphs')


args = parser.parse_args()




# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Download the dataset
train_flag = True
batch_size = 500


trainset = CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=train_flag)
n = len(trainset)
images = [preprocess(x[0]) for x in trainset]
labels = [x[1] for x in trainset]

features = []
all_image_features = None
with torch.no_grad():
    for k in trange((len(images) // batch_size) + 1):
        L = k * batch_size 
        R = min(n , (k+1) * batch_size)
        if L >= R:
            continue
        data = torch.stack(images[L : R]).to(device)
        image_features = model.encode_image(data)
        features.append(image_features.cpu().numpy())

features = np.concatenate(features)
if train_flag:
    np.savez(os.path.join(args.store_dir , 'cifar10-train2'), feature=features, label=labels)
else:
    np.savez(os.path.join(args.store_dir ,'cifar10-test')  , feature=features, label=labels)
