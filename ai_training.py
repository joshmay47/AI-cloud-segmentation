# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 15:13:22 2022

@author: Josh, Mertash

This file was used for when the training set and testing set is broken up into 
individual TCs and years and basins.
"""
import numpy as np 
import os
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import time
from processor import iou, confusionMatrix, convertImages, selectionsTo3
import pickle
from glob import glob

directory = ""

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

n_epochs = 20
lr = 2e-4
batch_size = 12
temporal_distance = 3
train_years = ["2001","2002","2003","2004"]
train_basin = "*" # set to "*" for all basins
test_years = ["2005"]
test_basin = "*" # set to "*" for all basins

#Set device to GPU_indx if GPU is avaliable
GPU_indx = 0
device = torch.device(GPU_indx if torch.cuda.is_available() else 'cpu')
print(device)

loadedModel = False
if temporal_distance is not None:
    in_channels = 3
else:
    in_channels = 1
    
class tcFiles():
    def __init__(self, years, basin, training = True, partition = 80, temporal_distance=None):
        # The first {partition}% is used for training, the rest used for testing
        self.temporal_distance = temporal_distance
        
        tcFileNames = []
        for year in years:
            tcFileNames += [file for file in sorted(glob(f"Satellites/{year}/{basin}/*.pkl"))]
        
        if training:
            self.tcFileNames = tcFileNames[:int(len(tcFileNames)*(partition/100))]
        else:
            self.tcFileNames = tcFileNames[int(len(tcFileNames)*(partition/100)):]

    #Returns images and labels corresponding to the [index]s TC
    def __getitem__(self, index):
        with open(self.tcFileNames[index],"rb") as pkl_file:
            images,labels,selections = pickle.load(pkl_file)
        if self.temporal_distance is not None:
            images = selectionsTo3(labels,images,selections,temporalDistance=self.temporal_distance)
            selections[:self.temporal_distance] = False
            selections[-self.temporal_distance:] = False
        else:
            images = images[selections] # selectionsTo3() normally slices images
            
        labels = labels[selections]
        #Make image in range (-1,1) and both in float32 as torch requires
        images = convertImages(images)
        labels = labels.astype(np.float32)

        return images, labels
    
    def __len__(self):
        return len(self.tcFileNames)
    
class satelliteDataset(Dataset):
    def __init__(self, data1,data2):
        self.images = data1
        self.labels = data2
        self.to_tensor = torchvision.transforms.ToTensor()
    def __getitem__(self, index):
        image = self.images[index]
        image = self.to_tensor(image)
        label = self.labels[index]
        label = self.to_tensor(label).type(torch.LongTensor).squeeze(0)
        return image,label
    def __len__(self):
        return len(self.images)
    
class compactDataset(Dataset):
    def __init__(self, data1):
        self.images = data1
        self.to_tensor = torchvision.transforms.ToTensor()
    def __getitem__(self, index):
        image = self.images[index]
        image = self.to_tensor(image)
        return image
    def __len__(self):
        return len(self.images)

# Unet0
class Unet0(nn.Module):
    def __init__(self):
        super(Unet0, self).__init__()
        
        self.down1 = Unetdown(in_channels,16,True)
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
        x1 = self.down1(x)      # x1 = 16 x 512 x 512
        x2 = self.down2(x1)     # x2 = 32 x 256 x 256
        x3 = self.down3(x2)     # x3 = 64 x 128 x 128
        x4 = self.down4(x3)     # x4 = 128 x 64 x 64
        x5 = self.down5(x4)     # x5 = 256 x 32 x 32
        x6 = self.down6(x5)     # x6 = 512 x 16 x 16
        
        x7 = self.up7(x6,x5)    # x7 = 256 x 32 x 32
        x8 = self.up8(x7,x4)    # x8 = 128 x 64 x 64
        x9 = self.up9(x8,x3)    # x9 = 64 x 128 x 128
        x10 = self.up10(x9,x2)  # x10 = 32 x 256 x 256
        x11 = self.up11(x10,x1) # x11 = 1 x 512 x 512
        
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

#Transfer model to GPU
model = Unet0().to(device)
# Find files relevant to training and testing
dataset_train = tcFiles(train_years,train_basin,True,partition = 100, temporal_distance=temporal_distance)
dataset_test = tcFiles(test_years,test_basin,False, partition = 0, temporal_distance=temporal_distance)
print(f"Training with {len(dataset_train.tcFileNames)} TCs.")
print(f"Testing with {len(dataset_test.tcFileNames)} TCs.")


def train(model_save_name):
    print(model)

    #Use an Adam optimiser to update the weights of the model
    optimiser = optim.Adam(model.parameters(), lr = lr)

    #Cross entropy - softmax over the two classes and negative log liklihood loss
    loss_fn = nn.CrossEntropyLoss()


    #Set maximum epochs and create empty lists to store losses
    Train_loss = []
    Test_loss = []

    for epoch in range(n_epochs):
        running_loss_train = 0.0
        running_loss_test = 0.0
        start_time = time.time()
        
        
        # Train:
        model.train()
        length = 0
        for tcI, (images,labels) in enumerate(dataset_train): # Iterate through training TCs
            convertedDataset = satelliteDataset(images,labels)
            data_loader_train = DataLoader(dataset=convertedDataset, batch_size=batch_size, shuffle=True)
            for i, (image,label) in enumerate(data_loader_train): # Iterate through image, label pairs in TC (in batches)
                image = image.to(device)
                label = label.to(device)
                
                #Forward pass through model
                outputs = model(image)
                
                #Compute cross entropy loss
                loss = loss_fn(outputs, label)
                running_loss_train += loss.item()

                #Gradients are accumulated, so they should be zeroed before calling backwards
                optimiser.zero_grad()
                
                #Backward pass through model and update the model weights
                loss.backward()
                optimiser.step()
            length += i # Used to calculate how many frames in training set
        
        if epoch == 0:
            training_size = length*batch_size
            print(f"Trained on ~{training_size} image, label pairs.")
        running_loss_train /= length
        Train_loss.append(running_loss_train)
        
        # Test:
        model.eval()
        with torch.no_grad():
            length = 0
            for tcI, (images,labels) in enumerate(dataset_test):
                convertedDataset = satelliteDataset(images, labels)
                data_loader_test = DataLoader(dataset=convertedDataset, batch_size=batch_size, shuffle=False)
                for i, (image, label) in enumerate(data_loader_test):   
                    image = image.to(device)
                    label = label.to(device)
    
                    outputs = model(image)
                    loss = loss_fn(outputs, label)
                    running_loss_test += loss.item()
                length += i
        
        if epoch == 0:
            testing_size = length*batch_size
            print(f"Tested on ~{testing_size} image, label pairs.")
        running_loss_test /= length
        Test_loss.append(running_loss_test)
        
        end_time = time.time()
        
        print(f'[Epoch {epoch:0>2}] Train Loss: {running_loss_train:.4f}, Val Loss: {running_loss_test:.4f}, Time: {end_time-start_time:.1f}s')
        torch.save(model.state_dict(),f"{model_save_name}Epoch{epoch:0>2}.pt")
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.plot(Train_loss, '-', label = 'Training Loss')
    plt.plot(Test_loss, '-', label = 'Validation Loss')
    plt.grid(axis="both")
    plt.xticks(range(n_epochs))
    plt.legend()
    plt.show()


    data_loader_iter = iter(data_loader_test)

    with torch.no_grad():
        for i in range(10):
            try:
                image, label = next(data_loader_iter)
            except:
                break
            
            plt.subplot(1,3,1)
            plt.imshow(image[0,0,:,:], cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.xlabel("Base Image")
            
            image = image.to(device)
            output = model(image)
            pred = torch.argmax(output,dim=1,keepdim=True)
            
            plt.subplot(1,3,2)
            plt.imshow(label[0,:,:], cmap='gray')
            acc = np.round(100*iou(pred.cpu().numpy()[0,0,:,:],label[0,:,:]),2)
            plt.title(f"IoU = {acc}%")
            plt.xlabel("Ground Truth")
            plt.xticks([])
            plt.yticks([])
            
            plt.subplot(1,3,3)
            plt.imshow(pred.cpu().numpy()[0,0,:,:], cmap='gray')
            plt.xlabel("Prediction")
            plt.xticks([])
            plt.yticks([])
            plt.show()
    
    return Train_loss, Test_loss

def predict(images):
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
            predictions.append(np.array(torch.argmax(model(image),dim=1,keepdim=True).cpu()[0,0,:,:]))
    if ndims == 2:
        return np.array(predictions, dtype = np.uint8)[0]
    else: 
        return np.array(predictions, dtype = np.uint8)

def meanIOU(part):
    dataset_test = SatDataset(data_path,False, partition = part)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=1, shuffle=False)
    
    sumIOU=0
    for i, (image,label) in enumerate(data_loader_test):
        pred = torch.argmax(model(image.to(device)),dim=1,keepdim=True).cpu().numpy()[0,0,:,:]
        lab = label.to(device).cpu().numpy()[0,:,:]
        ioui = iou(pred,lab)
        sumIOU += ioui
    return sumIOU/i

def meanConfusionMatrix(part):
    dataset_test = SatDataset(data_path,False, partition = part)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=1, shuffle=False)

    sumconfusionMatrix=np.zeros((2,2))
    for i, (image,label) in enumerate(data_loader_test):
        pred = torch.argmax(model(image.to(device)),dim=1,keepdim=True).cpu().numpy()[0,0,:,:]
        lab = label.to(device).cpu().numpy()[0,:,:]
        confusionMatrixi = confusionMatrix(pred,lab)
        sumconfusionMatrix += confusionMatrixi
    print(f"Mean IOU: {sumconfusionMatrix[1,1]/(i-sumconfusionMatrix[0,0])}")
    print(f"Confusion Matrix:\n{100*sumconfusionMatrix.round(4)/i}")

def loadModel(filename):
    global loadedModel
    loadedModel = True
    model.load_state_dict(torch.load(f"{directory}{filename}.pt"))
    model.eval()
    
def showPerformance(frameNo):
    if not loadedModel:
        print("WARNING: MODEL HAS NOT BEEN LOADED, TYPE loadModel() OR IGNORE IF NOT APPLICABLE")
    prediction = predictSingle(satellite[frameNo])
    fig = plt.figure(figsize=(18,6))
    plt.suptitle(f"Frame:{frameNo}   IoU={int(100*iou(prediction, processed[frameNo]))}%", fontsize = 25.0)
    ax1 = fig.add_subplot(131)
    ax1.set_title("Satellite", fontsize=20.0)
    ax1.axis("off")
    ax2 = fig.add_subplot(132)
    ax2.set_title("Ground Truth", fontsize = 20.0)
    ax2.axis("off")
    ax3 = fig.add_subplot(133)
    ax3.set_title("Prediction", fontsize = 20.0)
    ax3.axis("off")
    
    ax1.imshow(satellite[frameNo])
    ax2.imshow(processed[frameNo], cmap = "gray")
    ax3.imshow(prediction, cmap="gray")
    
    plt.show()
