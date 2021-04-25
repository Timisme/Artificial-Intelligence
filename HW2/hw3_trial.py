import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import models
from torch import optim
from torchsummary import summary
import os, torch, torchvision, random
import pandas as pd
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from VGGmodel import BuildModel
from customdataset import DogDataset
import time


normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

# Transformer
train_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
 
test_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

# 1.2. 填入 ??? 的部份

def split_Train_Val_Data(data_dir):
    
    dataset = ImageFolder(data_dir) 
    
    # 建立 20 類的 list
    character = [[] for i in range(len(dataset.classes))]
    # print(character)
    
    # 將每一類的檔名依序存入相對應的 list
    for x, y in dataset.samples:
        character[y].append(x)
      
    train_inputs, test_inputs = [], []
    train_labels, test_labels = [], []
    
    for i, data in enumerate(character): # 讀取每個類別中所有的檔名 (i: label, data: filename)
        
        np.random.seed(42)
        np.random.shuffle(data)
            
        # -------------------------------------------
        # 將每一類都以 8:2 的比例分成訓練資料和測試資料
        # -------------------------------------------
        
        num_sample_train = 0.8
        num_sample_test = 0.2
        
        # print(str(i) + ': ' + str(len(data)) + ' | ' + str(num_sample_train) + ' | ' + str(num_sample_test))
        
        for x in data[:int(len(data)*num_sample_train)] : # 前 80% 資料存進 training list
            train_inputs.append(x)
            train_labels.append(i)
            
        for x in data[int(len(data)*num_sample_train):] : # 後 20% 資料存進 testing list
            test_inputs.append(x)
            test_labels.append(i)

    train_dataloader = DataLoader(DogDataset(train_inputs, train_labels, train_transformer),
                                  batch_size = batch_size, shuffle = True, num_workers= 2)
    test_dataloader = DataLoader(DogDataset(test_inputs, test_labels, test_transformer),
                                  batch_size = batch_size, shuffle = False, num_workers=2)
 
    return train_dataloader, test_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

batch_size = 4
lr = 1e-3
epochs = 20

data_dir = 'stanford_dog'
# 2.3. 

train_dataloader, test_dataloader = split_Train_Val_Data(data_dir)

C = BuildModel().to(device) # 使用內建的 model 或是自行設計的 model
optimizer_C = optim.SGD(C.parameters(), lr = lr) # 選擇你想用的 optimizer

# print(summary(C, (3,224,224)) # 利用 torchsummary 的 summary package 印出模型資訊，input size: (3 * 224 * 224)

# Loss function
criterion = nn.CrossEntropyLoss()
 # 選擇想用的 loss function
loss_epoch_C = []
train_acc, test_acc = [], []
best_acc, best_auc = 0.0, 0.0

if __name__ == '__main__':    
    
    for epoch in range(epochs):
    
        iter = 0
        correct_train, total_train = 0, 0
        correct_test, total_test = 0, 0
        train_loss_C = 0.0

        C.train() # 設定 train 或 eval
      
        print('epoch: ' + str(epoch + 1) + ' / ' + str(epochs))  
        
        # ---------------------------
        # Training Stage
        # ---------------------------
        
        for i, (x, label) in enumerate(train_dataloader) :
                     
            x, label = x.to(device), label.to(device)
            label=torch.tensor(label, dtype=torch.long)             
            optimizer_C.zero_grad()
            
            outputs = C(x) # 將訓練資料輸入至模型進行訓練
            # print('outputs:',len(outputs),'label:',len(label))
            # print('outputs:',outputs,'label:',label)
            # print(label.size())
            loss = criterion(outputs, label) # 計算 loss
            
            loss.backward() # 將 loss 反向傳播
            optimizer_C.step() # 更新權重
            
            # 計算訓練資料的準確度 (correct_train / total_train)
            _, predicted = torch.max(outputs.data,1)
            total_train += label.size(0)
            correct_train += (predicted == label).sum().item()

            train_loss_C += loss.item()
            iter += 1
                    
        print('Training epoch: %d / loss_C: %.3f | acc: %.3f' % \
              (epoch + 1, train_loss_C / iter, correct_train / total_train))

        
        # --------------------------
        # Testing Stage
        # --------------------------
        
        C.eval() # 設定 train 或 eval
          
        for i, (x, label) in enumerate(test_dataloader) :
          
            with torch.no_grad(): # 測試階段不需要求梯度
                x, label = x.to(device), label.to(device)
                
                outputs = C(x) # 將測試資料輸入至模型進行測試
#                 ??? # 計算測試資料的準確度
                _, predicted = torch.max(outputs.data,1)
                total_test += label.size(0)
                correct_test += (predicted == label).sum().item()
        
        print('Testing acc: %.3f' % (correct_test / total_test))
                                     
        train_acc.append(100 *(correct_train / total_train)) # training accuracy
        test_acc.append(100 * (correct_test / total_test))  # testing accuracy
        loss_epoch_C.append(train_loss_C / iter) # loss 


