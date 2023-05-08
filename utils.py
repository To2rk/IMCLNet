import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms
import torch
from torch.utils.data import Dataset
import os, sys
import cv2
import random
from prettytable import PrettyTable

labels_MalImg = ['Adialer.C', 'Agent.FYI', 'Allaple.A', 'Allaple.L', 'Alueron.gen!J', 'Autorun.K', 'C2LOP.gen!g', 
          'C2LOP.P', 'Dialplatform.B', 'Dontovo.A', 'Fakerean', 'Instantaccess', 'Lolyda.AA1', 'Lolyda.AA2', 
          'Lolyda.AA3', 'Lolyda.AT', 'Malex.gen!J', 'Obfuscator.AD', 
          'Rbot!gen', 'Skintrim.N', 'Swizzor.gen!E', 'Swizzor.gen!I', 'VB.AT', 'Wintrim.BX', 'Yuner.A']

labels_BIG2015 = ['Ramnit', 'Lollipop', 'Kelihos_ver3', 'Vundo', 'Simda', 'Tracur', 'Kelihos_ver1', 'Obfuscator.ACY', 'Gatak']


def chooseDataset(dataset):
    if dataset == 'MalImg':
        labels = labels_MalImg
    elif dataset == 'BIG2015':
        labels = labels_BIG2015
    else:
        print('Dataset load error, please check!')
        sys.exit()

    return labels

class MalwareImageDataset(Dataset):
    def __init__(self, dataset_path, dataset):

        self.dataset_path = dataset_path
        self.dataset = dataset
        self.images = os.listdir(self.dataset_path)

        self.transform = transforms.Compose([
            transforms.ToTensor(),    
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_index = self.images[index]
        img_path = os.path.join(self.dataset_path, image_index)

        image = self.transform(cv2.imread(img_path))
        labels = chooseDataset(self.dataset)
        label = labels.index(image_index.split('-')[0])

        return image, label

class ConfusionMatrix(object):

    def __init__(self, num_classes: int, labels_list: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels_list = labels_list

    def update(self, pred, label):

        for p, t in zip([pred], [label]):
            self.matrix[p, t] += 1

    def summary(self):

        Precision_list = []
        Recall_list = []
        F1_Score_list = []

        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "F1 Score"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN

            Accuracy = round((TP + TN) / (TP + TN + FN+ FP), 5) if TP + TN + FN+ FP != 0 else 0.
            Precision = round(TP / (TP + FP), 5) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 5) if TP + FN != 0 else 0.
            F1_Score = round((2 * Precision * Recall) / (Precision + Recall), 5) if Precision + Recall != 0 else 0.

            Precision_list.append(Precision)
            Recall_list.append(Recall)
            F1_Score_list.append(F1_Score)

            table.add_row([self.labels_list[i], Precision, Recall, F1_Score])

        macro_Precision = round(np.mean(Precision_list), 5)
        macro_Recall = round(np.mean(Recall_list), 5)
        macro_F1_Score = round(np.mean(F1_Score_list), 5)
        
        table.add_row(['Macro', macro_Precision, macro_Recall, macro_F1_Score])

        print(table)

        return str(macro_Precision)

    def plot(self, ConfusionMatrixPath):

        params = {
            # 'figure.figsize': '9.5, 8', # BIG2015
            # 'font.size': 14,               # BIG2015
            
            'figure.figsize': '9.75, 8', # MalImg
            'font.size': 10,               # MalImg
            'figure.autolayout': True,
            'savefig.dpi' : 1200,
            'figure.dpi' : 1200,
        }
        plt.rcParams.update(params)

        matrix = self.matrix
        plt.imshow(matrix, cmap=plt.cm.Blues)

        plt.xticks(range(self.num_classes), self.labels_list, rotation=90)
        plt.yticks(range(self.num_classes), self.labels_list)
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')

        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.savefig(ConfusionMatrixPath)

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_num, trainable_num
