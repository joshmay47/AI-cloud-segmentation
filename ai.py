# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 15:13:22 2022

@author: Josh, Mertash
"""
import numpy as np 
import os
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from numba import vectorize

unetUsed = "ai_statedict"
directory = ""
inChannels = 3

class compactDataset(Dataset):
    def __init__(self, data):
        self.images = data
        self.to_tensor = torchvision.transforms.ToTensor()
    def __getitem__(self, index):
        image = self.images[index]
        image = self.to_tensor(image)
        return image
    def __len__(self):
        return len(self.images)
    
# UNET
class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__()
        
        self.down1 = Unetdown(inChannels,16,True)
        self.down2 = Unetdown(16,32,False)
        self.down3 = Unetdown(32,64,False)
        self.down4 = Unetdown(64,128,False)
        self.down5 = Unetdown(128,256,False)
        self.down6 = Unetdown(256,512,False)
        
        self.up7 = Unetup(512,256,False)
        self.up8 = Unetup(256,128,False)
        self.up9 = Unetup(128,64,False)
        self.up10 = Unetup(64,32,False)
        self.up11 = Unetup(32,16,True)
    
    def forward(self, x):
        # x = 1 x 512 x 512
        x1 = self.down1(x)
        # x1 = 16 x 512 x 512
        x2 = self.down2(x1)
        # x2 = 32 x 256 x 256
        x3 = self.down3(x2)
        # x3 = 64 x 128 x 128
        x4 = self.down4(x3)
        # x4 = 128 x 64 x 64
        x5 = self.down5(x4)
        # x5 = 256 x 32 x 32
        x6 = self.down6(x5)
        # x6 = 512 x 16 x 16
        
        x7 = self.up7(x6,x5)
        # x7 = 256 x 32 x 32
        x8 = self.up8(x7,x4)
        # x8 = 128 x 64 x 64
        x9 = self.up9(x8,x3)
        # x9 = 64 x 128 x 128
        x10 = self.up10(x9,x2)
        # x10 = 32 x 256 x 256
        x11 = self.up11(x10,x1)
        # x11 = 1 x 512 x 512
        
        return x11

class Unetdown(nn.Module):
    def __init__(self, input_nc, output_nc, first_layer = False):
        super(Unetdown, self).__init__()
        
        model = []
        if not first_layer:
            model += [nn.MaxPool2d(2)]
        
        model += [nn.Conv2d(input_nc, output_nc, 3, padding=1),
                  nn.ReLU(),
                  nn.Conv2d(output_nc, output_nc, 3, padding=1),
                  nn.ReLU()]
        
        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        out = self.model(x)
        
        return out

class Unetup(nn.Module):
    def __init__(self, input_nc, output_nc, last_layer = False):
        super(Unetup, self).__init__()

        self.up= nn.ConvTranspose2d(input_nc, output_nc, 2, stride=2)
        model = []
        model += [nn.Conv2d(input_nc, output_nc, 3, padding=1),
                  nn.ReLU(),
                  nn.Conv2d(output_nc, output_nc, 3, padding=1),
                  nn.ReLU()]
        
        if last_layer:
            model += [nn.Conv2d(output_nc, 2, 1, padding=0)]
          
        self.model = nn.Sequential(*model)
            
    def forward(self, x1, x2):
        x1 = self.up(x1)
        out = self.model(torch.cat([x1,x2],dim=1))
        
        return out

def predict(images):
    ndims = images.ndim
    if inChannels == 1:
        singularImageDims = 2
    else:
        singularImageDims = 3
    if ndims == singularImageDims:
        images = np.expand_dims(images, axis=0)
    images = convertImages(images)
    predictions = []
    images = compactDataset(images)
    images = iter(DataLoader(images, batch_size = 1))
    with torch.no_grad():
        for image in images:
            image = image.to(device)
            predictions.append(np.array(torch.argmax(model(image),dim=1,keepdim=True).cpu()[0,0,:,:]))
    if ndims == singularImageDims:
        return np.array(predictions, dtype = np.uint8)[0]
    else: 
        return np.array(predictions, dtype = np.uint8)
    
def predictConfidence(images):
    ndims = images.ndim
    if ndims == 2:
        images = np.expand_dims(images, axis=0)
    images = convertImages(images)
    predictions = []
    images = compactDataset(images)
    images = iter(DataLoader(images, batch_size = 1))
    with torch.no_grad():
        for image in images:
            image = image.to(device)
            pred = np.array(model(image).cpu())
            prediction = confidence(pred[:,0,:,:], pred[:,1,:,:])
            predictions.append(prediction)
    
    return np.squeeze(np.array(predictions, dtype = np.float32))

@vectorize("float32(uint8)")
def convertImages(pixel):
    return (pixel-128)/128

@vectorize("float32(float32,float32)")
def confidence(negativeScore, positiveScore):
    return (positiveScore-negativeScore)/(positiveScore+negativeScore)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device(0 if torch.cuda.is_available() else 'cpu')
model = UNET().to(device)
model.load_state_dict(torch.load(f"{directory}{unetUsed}.pt"))
model.eval()
params = f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters"