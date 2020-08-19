import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from main_lib import *

###############################################################################
##              tests
###############################################################################
def accuracy(model, test_loader):
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        totalClass = np.array([0,0,0,0,0])
        correctClass = np.array([0,0,0,0,0])
        for images, labels in test_loader:
            images = images.view(-1, 2500, 12)  # reshape x to (batch, time_step, input_size)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for lab, pred in zip(labels, predicted):
                totalClass[lab] += 1
                correctClass[lab] += (lab == pred)

        accurArr = correctClass/totalClass
        print('Test Accuracy of the model: {} %'.format((correct / total) * 100))
        print(accurArr)

def accuracy_for_classes(model, test_loader):
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        totalClass = np.array([0,0,0,0,0])
        correctClass = np.array([0,0,0,0,0])
        af = np.array([0,0,0,0,0])
        sa = np.array([0,0,0,0,0])
        afTotal = 0
        saTotal = 0
        for images, labels in test_loader:
            images = images.view(-1, 2500, 12)  # reshape x to (batch, time_step, input_size)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for lab, pred in zip(labels, predicted):
                totalClass[lab] += 1
                correctClass[lab] += (lab == pred)

                if lab == 3:
                    afTotal += 1
                    af[pred] += 1
                if lab == 4:
                    saTotal += 1
                    sa[pred] += 1

        accurArr = correctClass/totalClass
        print('Test Accuracy of the model: {} %'.format((correct / total) * 100))
        print(accurArr)
        print('\n')
        print('Atrial Fibrillation classes: {}'.format(af / afTotal))
        print('Sinus Arhytmia classes: {}'.format(sa / saTotal))

def calc_prec_rec(model, test_loader, class1_idx, class2_idx):
    # Calculate precision and recall
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        totalClass = np.array([0,0,0,0,0])
        correctClass = np.array([0,0,0,0,0])
        af = np.array([0,0,0,0,0])
        sa = np.array([0,0,0,0,0])
        class1_Total = 0
        class2_Total = 0
        class1_TP = 0
        class1_TN = 0
        class1_FP = 0
        class1_FN = 0
        class2_TP = 0
        class2_TN = 0
        class2_FP = 0
        class2_FN = 0
        for images, labels in test_loader:
            #images = images.view(-1, 2500, 12)  # reshape x to (batch, time_step, input_size)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for lab, pred in zip(labels, predicted):
                class1_TP += (lab == class1_idx) and (pred == class1_idx)
                class1_TN += (lab == class2_idx) and (pred == class2_idx)
                class1_FP += (lab == class2_idx) and (pred == class1_idx)
                class1_FN += (lab == class1_idx) and (pred != class1_idx)

                class2_TP += (lab == class2_idx) and (pred == class2_idx)
                class2_TN += (lab == class1_idx) and (pred == class1_idx)
                class2_FP += (lab == class1_idx) and (pred == class2_idx)
                class2_FN += (lab == class2_idx) and (pred != class2_idx)

        precision1 = class1_TP.item() / (class1_TP.item() + class1_FP.item())
        recall1 = class1_TP.item() / (class1_TP.item() + class1_FN.item())

        precision2 = class2_TP.item() / (class2_TP.item() + class2_FP.item())
        recall2 = class2_TP.item() / (class2_TP.item() + class2_FN.item())

        F1Score_1 = 2*precision1*recall1 / (precision1 + recall1)
        F1Score_2 = 2 * precision2 * recall2 / (precision2 + recall2)

        print('Test Accuracy of the model: {} %'.format((correct / total) * 100))

        print('class #1, precision {}, recall {}, f1 score {}'.format(precision1, recall1, F1Score_1))
        print('class #2, precision {}, recall {}, f1 score {}'.format(precision2, recall2, F1Score_2))


####################################################################################################
##              Main
####################################################################################################
PATH = './Model/'
RNN_NAME = 'conv_net_model_rnn_2.ckpt'
CNN_NAME = 'conv_net_model_2class.ckpt'
BATCH_SIZE = 30

model = ConvNet()
model.load_state_dict(torch.load(PATH + CNN_NAME))

test_dataset = EkgDataSetForTwo(0, '../ReadyDataAug (copy)/', 2, 4)

# Data loader
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

calc_prec_rec(model, test_loader, 2, 4)
