import os
import numpy as np
import PIL
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torch
from training import get_dataloader
from training import Net
import torchvision.models as models
from test import test


## TODO: Specify model architecture 
VGG16 = models.vgg16(pretrained=True)
model_transfer = VGG16

for param in model_transfer.parameters():
    param.requires_grad = False
for param in model_transfer.classifier.parameters():
    param.requires_grad = True
    
num_ftrs = model_transfer.classifier[6].in_features
model_transfer.classifier[6] = nn.Linear(num_ftrs, 133)
use_cuda = False
if use_cuda:
    model_transfer = model_transfer.cuda()


if __name__ == '__main__':
    batch_size = 10
    
    print ('use_cuda: {}'.format(use_cuda))
    loaders_scratch = get_dataloader(batch_size, 256, 224)
    criterion_transfer = nn.CrossEntropyLoss()
    model_transfer.load_state_dict(torch.load('model_transfer.pt'))
    output = model(data)
    test(loaders_scratch, model_transfer, criterion_transfer, use_cuda)
