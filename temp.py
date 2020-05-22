import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    ### TODO: choose an architecture, and complete the class
    def __init__(self):
        super(Net, self).__init__()
        ## Define layers of a CNN
        self.conv1 = nn.Conv2d(3, 3, kernel_size=7, stride=2, padding=1)
        ## (-1, 3, 110, 110)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=5, stride=2, padding=1)
        ## (-1, 3, 54, 54)
        self.conv3 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        ## (-1, 6, 26, 26)
        self.conv4 = nn.Conv2d(6, 9, kernel_size=3, stride=2, padding=1)
        ## (-1, 9, 13, 13)
        self.fc1 = nn.Linear(9 * 13 * 13, 1000)
        self.fc2 = nn.Linear(1000, 500)  
        self.fc3 = nn.Linear(500, 133)
    
    def forward(self, x):
        ## Define forward behavior
        x = x.view((-1, 3, 224, 224))
        x = F.relu(self.conv1(x))
        # print (x.shape)   # (-1, 3, 110, 110)
        x = F.relu(self.conv2(x))
        # print (x.shape)   # (-1, 3, 54, 54)
        x = F.relu(self.conv3(x))
        # print (x.shape)   # (-1, 6, 26, 26)
        x = F.relu(self.conv4(x))
        # print (x.shape)   # (-1, 9, 13, 13)
        fc_input_dim = x.size(1) * x.size(2) * x.size(3)
#         print (fc_input_dim)
        x = x.view(-1, fc_input_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#-#-# You so NOT have to modify the code below this line. #-#-#

# instantiate the CNN
model_scratch = Net()

# move tensors to GPU if CUDA is available
if use_cuda:
    model_scratch.cuda()



#####################################################################  2

import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    ### TODO: choose an architecture, and complete the class
    def __init__(self):
        super(Net, self).__init__()
        ## Define layers of a CNN
        self.conv0 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(6, 9, kernel_size=5, stride=2, padding=1)
        self.conv3 = nn.Conv2d(9, 12, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(12 * 7 * 7, 200)
        self.fc2 = nn.Linear(200, 500)  #output layer
        self.fc3 = nn.Linear(500, 133)
        self.drop = nn.Dropout(0.5)
    
    def forward(self, x):
        ## Define forward behavior
        x = x.view(-1, 3, 64, 64)
        x = self.pool(F.relu(self.conv0(x)))
        # print(x.shape) 
        x = F.relu(self.conv1(x))
        # print (x.shape)   # -1, 6, 30, 30
        x = F.relu(self.conv2(x))
        # print (x.shape)   # -1, 9, 14, 14
        x = F.relu(self.conv3(x))
        # print (x.shape)   # -1, 12, 7, 7
        fc_input_dim = x.size(1) * x.size(2) * x.size(3)
        # print (fc_input_dim)
        x = x.view(-1, fc_input_dim)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x

#-#-# You so NOT have to modify the code below this line. #-#-#

# instantiate the CNN
model_scratch = Net()

# move tensors to GPU if CUDA is available
if use_cuda:
    model_scratch.cuda()





import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    ### TODO: choose an architecture, and complete the class
    def __init__(self):
        super(Net, self).__init__()
        ## Define layers of a CNN
        self.conv1 = nn.Conv2d(3, 8, 5, padding=2)
        self.conv2 = nn.Conv2d(8, 16, 5, padding=2)
        self.conv3 = nn.Conv2d(16, 32, 5, padding=2)
        self.conv4 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv5 = nn.Conv2d(64, 64, 5, padding=2)
        # self.conv6 = nn.Conv2d(64, 128, 3, padding=1)
        # self.conv7 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(4096, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 133)
        self.drop = nn.Dropout(0.5)

        

    
    def forward(self, x):
        ## Define forward behavior
        x = x.view(-1, 3, 256, 256)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        # print (x.shape)   # -1, 64, 8, 8
        fc_input_dim = x.size(1) * x.size(2) * x.size(3)
        # print (fc_input_dim)
        x = x.view(-1, fc_input_dim)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x

#-#-# You so NOT have to modify the code below this line. #-#-#

# instantiate the CNN
model_scratch = Net()

# move tensors to GPU if CUDA is available
if use_cuda:
    model_scratch.cuda()


##################################################################
import os
import numpy as np
import PIL
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torch

def get_dataloader(batch_size, image_size):
    
    data_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(
        root='dog_images/train',
        transform=data_transform
    )
    valid_dataset = datasets.ImageFolder(
        root='dog_images/valid',
        transform=data_transform
    )
    test_dataset = datasets.ImageFolder(
        root='dog_images/test',
        transform=data_transform
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
        self.conv0 = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=1)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=1)
        self.fc1 = nn.Linear(24 * 3 * 3, 200)
        self.fc2 = nn.Linear(200, 500)  #output layer
        self.fc3 = nn.Linear(500, 133)
        self.drop = nn.Dropout(0.3)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        ## Define forward behavior
        x = x.view(-1, 3, 130, 130)
        x = self.pool(F.relu(self.conv4(x)))
        # print(x.shape, "expect -1, 3, 64, 64")
        x = self.pool(F.relu(self.conv0(x)))
        # print(x.shape, "expect -1, 3, 31, 31")
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.shape, "expect -1, 6, 14, 14")   
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape, "expect -1, 12, 7, 7")    
        x = self.pool(F.relu(self.conv3(x)))
        # print(x.shape, "expect -1, 24, 3, 3")   
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
        print('Minimum validation loss : {}'.format(valid_loss_min))
        
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
    loaders_scratch = get_dataloader(batch_size, 130)
    model_scratch = Net()
    if use_cuda:

        model_scratch.cuda()

    #test block:
    for _, sample in enumerate(loaders_scratch['train']):
        test = model_scratch(sample[0].cuda())
        print(test.shape)
        break

    criterion_scratch = nn.CrossEntropyLoss()
    optimizer_scratch = optim.Adam(model_scratch.parameters(), lr=0.07)

    # train the model
    model_scratch = train(100, loaders_scratch, model_scratch, optimizer_scratch, 
                      criterion_scratch, use_cuda, 'model_scratch.pt')

    # load the model that got the best validation accuracy
    # model_scratch.load_state_dict(torch.load('model_scratch.pt'))