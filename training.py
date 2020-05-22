import os
import numpy as np
import PIL
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torch

def get_dataloader(batch_size, image_size, cropsize):


    transform = {
        'train': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(cropsize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )]
        ),
        'test': transforms.Compose([
            transforms.Resize((cropsize, cropsize)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )]
        )
    }

    


    train_dataset = datasets.ImageFolder(
        root='dog_images/train',
        transform=transform['train']
    )
    valid_dataset = datasets.ImageFolder(
        root='dog_images/valid',
        transform=transform['test']
    )
    test_dataset = datasets.ImageFolder(
        root='dog_images/test',
        transform=transform['test']
    )

    train_dataset_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    valid_dateset_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_dataset_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    loaders_scratch = {
        'train': train_dataset_loader,
        'valid': valid_dateset_loader,
        'test': test_dataset_loader
    }

    return loaders_scratch

#### neural network
class Net(nn.Module):
    ### TODO: choose an architecture, and complete the class
    def __init__(self):
        super(Net, self).__init__()
        ## Define layers of a CNN
        self.conv0 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=5, stride=2, padding=1)
        self.conv3 = nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(24 * 7 * 7, 200)
        self.fc2 = nn.Linear(200, 500)  #output layer
        self.fc3 = nn.Linear(500, 133)
        self.drop = nn.Dropout(0.3)
    
    def forward(self, x):
        ## Define forward behavior
        # print (x.shape)
        x = x.view(-1, 3, 64, 64)
        x = self.pool(F.relu(self.conv0(x)))
        # print(x.shape, "expect -1, 3, 32, 32")
        x = F.relu(self.conv1(x))
        # print(x.shape, "expect -1, 6, 30, 30")   
        x = F.relu(self.conv2(x))
        # print(x.shape, "expect -1, 9, 14, 14")    
        x = F.relu(self.conv3(x))
        # print(x.shape, "expect -1, 12, 7, 7")   
        fc_input_dim = x.size(1) * x.size(2) * x.size(3)
        x = x.view(-1, fc_input_dim)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    print (use_cuda)
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            ## record the average training loss, using something like
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            if batch_idx % 20 == 0:
                print (train_loss)
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            outputs = model(data)
            loss = criterion(outputs, target)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))


        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        ## TODO: save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            valid_loss_min = valid_loss
            torch.save(model.state_dict(), save_path)
        
        
    # return trained model
    return model


if __name__ == '__main__':

    batch_size = 64
    use_cuda = torch.cuda.is_available()
    print ('use_cuda: {}'.format(use_cuda))
    loaders_scratch = get_dataloader(batch_size, 80, 64)
    model_scratch = Net()
    if use_cuda:

        model_scratch.cuda()

    #test block:
    for _, sample in enumerate(loaders_scratch['train']):
        test = model_scratch(sample[0].cuda())
        print(test.shape)
        break

    criterion_scratch = nn.CrossEntropyLoss()
    optimizer_scratch = optim.Adam(model_scratch.parameters(), lr=0.001)

    # train the model
    model_scratch = train(100, loaders_scratch, model_scratch, optimizer_scratch, 
                      criterion_scratch, use_cuda, 'model_scratch.pt')

    # load the model that got the best validation accuracy
    # model_scratch.load_state_dict(torch.load('model_scratch.pt'))