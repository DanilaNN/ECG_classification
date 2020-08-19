import os
import json
import numpy as np
import pathlib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import LinearAxis, Range1d
from shutil import copyfile

class EkgDataSet(Dataset):

    def __init__(self, is_train, path):

        self.fullRoots = []
        self.allFiles = []
        self.filesCount = 0
        self.ekgDiagArr = []
        self.isTrain = is_train
        self.path = path
        self.path2Data = path + 'Train/' if self.isTrain else path + 'Test/'
        self.idxDiag = []

        tree = os.walk(self.path2Data)
        for root, dirs, files in tree:
            self.fullRoots.append(root)
            self.allFiles.append(files)
        self.allFiles = self.allFiles[1:]
        self.fullRoots = self.fullRoots[1:]

        # c = map(lambda x: x[0:800], self.allFiles)

        for i in range(len(self.allFiles)):
            self.idxDiag.append([i,self.fullRoots[i]])
            for j in range(len(self.allFiles[i])):
                self.ekgDiagArr.append([i, self.fullRoots[i]+'/'+self.allFiles[i][j]])

        print(self.idxDiag)

    def __getitem__(self, idx):
        with open(self.ekgDiagArr[idx][1]) as f:
            data = json.load(f)

        '''
        dataArr = []
        for key in data['Leads']:
            dataArr.append(data['Leads'][key]['Signal'])
        '''

        dataArr = np.array(data['Signal']).reshape(1, len(data['Signal']))

        pt_tensor_from_list = torch.FloatTensor(dataArr)

        pt_tensor_from_list = pt_tensor_from_list.unsqueeze(0)

        return [pt_tensor_from_list, self.ekgDiagArr[idx][0]]
        #return [pt_tensor_from_list, self.ekgDiagArr[idx][0]]

    def __len__(self):
        return len(self.ekgDiagArr)

    def bugCount(self):
        count = 0

        length = self.__len__()
        for i in range(length):
            with open(self.ekgDiagArr[i][1]) as f:
                data = json.load(f)

            if len(data['Leads']['avf']['Signal']) == 2500:
                count += 1


class EkgDataSetForTwo(Dataset):

    def __init__(self, is_train, path, ekg_1, ekg_2):

        self.fullRoots = []
        self.allFiles = []
        self.filesCount = 0
        self.ekgDiagArr = []
        self.isTrain = is_train
        self.path = path
        self.path2Data = path + 'Train/' if self.isTrain else path + 'Test/'
        self.idxDiag = []
        self.ekg_1 = ekg_1
        self.ekg_2 = ekg_2

        tree = os.walk(self.path2Data)
        for root, dirs, files in tree:
            self.fullRoots.append(root)
            self.allFiles.append(files)
        self.allFiles = self.allFiles[1:]
        self.fullRoots = self.fullRoots[1:]

        # c = map(lambda x: x[0:800], self.allFiles)

        for i in range(len(self.allFiles)):
            self.idxDiag.append([i, self.fullRoots[i]])
            for j in range(len(self.allFiles[i])):

                if i == self.ekg_1 or i == self.ekg_2:
                    self.ekgDiagArr.append([i, self.fullRoots[i]+'/'+self.allFiles[i][j]])

        print(self.idxDiag)

    def __getitem__(self, idx):
        with open(self.ekgDiagArr[idx][1]) as f:
            data = json.load(f)

        '''
        dataArr = []
        for key in data['Leads']:
            dataArr.append(data['Leads'][key]['Signal'])
        '''
        dataArr = data['Signal']

        pt_tensor_from_list = torch.FloatTensor(dataArr)

        return [pt_tensor_from_list.unsqueeze(0), self.ekgDiagArr[idx][0]]
        #return [pt_tensor_from_list, self.ekgDiagArr[idx][0]]

    def __len__(self):
        return len(self.ekgDiagArr)

    def bugCount(self):
        count = 0

        length = self.__len__()
        for i in range(length):
            with open(self.ekgDiagArr[i][1]) as f:
                data = json.load(f)

            if len(data['Leads']['avf']['Signal']) == 2500:
                count += 1



class RNN(nn.Module):
    def __init__(self, INPUT_SIZE):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=50,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(50, 2)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.MaxPool2d((1, 5), stride=(1, 5 )))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(1 * 60 * 64, 1000)
        self.fc2 = nn.Linear(1000, 2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

class ConvNet_2(nn.Module):
    def __init__(self):
        super(ConvNet_2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.MaxPool2d((1, 5), stride=(1, 5)))
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.MaxPool2d((1, 5), stride=(1, 5)))
        self.layer7 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.ReLU())
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.MaxPool2d((1, 10), stride=(1, 10)))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(1 * 20 * 256, 1000)
        self.fc2 = nn.Linear(1000, 2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        #out = out.reshape(out.size(0), -1)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

def test_rnn(model, test_loader):
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.view(-1, 300, 1)  # reshape x to (batch, time_step, input_size)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model: {} %'.format((correct / total) * 100))

def test_conv(model, test_loader):
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))
