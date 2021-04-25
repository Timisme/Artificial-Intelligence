{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Pytorch Homework 3: 利用 stanford dog (mini) dataset 來訓練一個分類器。\n",
    "\n",
    "## 作業說明\n",
    "\n",
    "這次的作業總共有 4 個部份，必須完成助教提供的 ipynb 檔，在檔案中某些區塊會有 ??? 處需要完成。\n",
    "\n",
    "0. 安裝需要使用的 package: pip install -r requirements.txt\n",
    "\n",
    "1. 實作 Dataloader\n",
    "    * 1.1. 實作一個可以讀取 stanford dog (mini) 的 Pytorch dataset。 ** (10%) **\n",
    "    * 1.2. 將每一個類別以 8 : 2 的比例分割成 Training data 和 Testing data 傳至 dataloader  ** (15%) **\n",
    "\n",
    "2. 建構 CNN model。\n",
    "    * 2.1. 利用 Pytorch 內建的 CNN model 來進行訓練。 ** (10%) **\n",
    "    * 2.2. 自行設計一個新的 CNN model 來進行訓練。 ** (20%) ** (至少達到 70% 以上的 Testing accuracy，否則只有一半的分數)\n",
    "    * 2.3. 利用 torchsummary 來印出上面兩個模型的架構資訊。 ** (5%) **\n",
    "\n",
    "3. 實作模型訓練和測試模型效能。 ** (30%) **\n",
    "4. 將每一個 epoch 的 Loss 以及 Training / Testing accuracy 紀錄下來並繪製成圖並儲存下來。 ** (10%) **\n",
    "    \n",
    "## 作業繳交\n",
    "\n",
    "* Deadline : 11/16 中午12:00        \n",
    "    * **遲交一天打 7 折**\n",
    "    * **遲交兩天打 5 折**\n",
    "    * **遲交三天以上不給予分數**\n",
    "    \n",
    "* 繳交方式 : 請將完成的 ipynb 檔 (分成兩個版本: 內建 model 版和自己設計的版本) 以及 Loss、Training / testing accuracy 的圖片，壓縮後上傳至 moodle。\n",
    "    * 建議先完成一個版本，然後將檔案複製後再完成另一個版本\n",
    "    \n",
    "* 壓縮檔內包含 :\n",
    "    * (你的學號)\\_(姓名)\\_HW3_2_1.ipynb  (E.g.F77777777_王小明_HW3_2_1.ipynb) \n",
    "    * (你的學號)\\_(姓名)\\_HW3_2_2.ipynb  (E.g.F77777777_王小明_HW3_2_2.ipynb) \n",
    "    * 兩個版本的 Loss.png、Acc.png (Loss_2_1.png、Loss_2_2.png、Acc_2_1.png、Acc_2_2.png)\n",
    "    * **格式不對的話會扣 10 分！！！**\n",
    "    \n",
    "* 有任何問題歡迎寄信至我的信箱\n",
    "    * 曹維廷 a0903511820@gmail.com"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os, torch, torchvision, random\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import torch.nn as nn\n",
    "import matplotlib.pyplot as plt\n",
    "from PIL import Image\n",
    "\n",
    "import torch.nn.functional as F\n",
    "import torchvision.transforms as transforms\n",
    "import torchvision.models as models\n",
    "from torch.utils.data import Dataset, random_split, DataLoader\n",
    "from torchvision.datasets import ImageFolder\n",
    "from torchvision import models\n",
    "from torch import optim\n",
    "from torchsummary import summary"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Part 1：Dataloader 實作"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "invalid syntax (<ipython-input-3-d7556b4286bd>, line 13)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  File \u001b[1;32m\"<ipython-input-3-d7556b4286bd>\"\u001b[1;36m, line \u001b[1;32m13\u001b[0m\n\u001b[1;33m    return ???\u001b[0m\n\u001b[1;37m           ^\u001b[0m\n\u001b[1;31mSyntaxError\u001b[0m\u001b[1;31m:\u001b[0m invalid syntax\n"
     ]
    }
   ],
   "source": [
    "# 1.1. 填入 ??? 的部份\n",
    "\n",
    "class DogDataset(Dataset):\n",
    "    \n",
    "    def __init__(self, filenames, labels, transform):\n",
    "        \n",
    "        self.filenames = filenames # 資料集的所有檔名\n",
    "        self.labels = labels # 影像的標籤\n",
    "        self.transform = transform # 影像的轉換方式\n",
    " \n",
    "    def __len__(self):\n",
    "        \n",
    "        return ??? # return DataSet 長度\n",
    " \n",
    "    def __getitem__(self, idx):\n",
    "        \n",
    "        image = Image.open(???).convert('RGB')\n",
    "        image = ??? # Transform image\n",
    "        label = np.array(???)\n",
    "                \n",
    "        return ???, ??? # return 模型訓練所需的資訊\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])\n",
    "\n",
    "# Transformer\n",
    "train_transformer = transforms.Compose([\n",
    "    transforms.Resize(256),\n",
    "    transforms.RandomResizedCrop(224),\n",
    "    transforms.RandomHorizontalFlip(),\n",
    "    transforms.ToTensor(),\n",
    "    normalize\n",
    "])\n",
    " \n",
    "test_transformer = transforms.Compose([\n",
    "    transforms.Resize(224),\n",
    "    transforms.CenterCrop(224),\n",
    "    transforms.ToTensor(),\n",
    "    normalize\n",
    "])\n",
    "\n",
    "# 1.2. 填入 ??? 的部份\n",
    "\n",
    "def split_Train_Val_Data(data_dir):\n",
    "    \n",
    "    dataset = ImageFolder(data_dir) \n",
    "    \n",
    "    # 建立 20 類的 list\n",
    "    character = [[] for i in range(len(dataset.classes))]\n",
    "    # print(character)\n",
    "    \n",
    "    # 將每一類的檔名依序存入相對應的 list\n",
    "    for x, y in dataset.samples:\n",
    "        character[???].append(???)\n",
    "      \n",
    "    train_inputs, test_inputs = [], []\n",
    "    train_labels, test_labels = [], []\n",
    "    \n",
    "    for i, data in enumerate(character): # 讀取每個類別中所有的檔名 (i: label, data: filename)\n",
    "        \n",
    "        np.random.seed(42)\n",
    "        np.random.shuffle(data)\n",
    "            \n",
    "        # -------------------------------------------\n",
    "        # 將每一類都以 8:2 的比例分成訓練資料和測試資料\n",
    "        # -------------------------------------------\n",
    "        \n",
    "        num_sample_train = ???\n",
    "        num_sample_test = ???\n",
    "        \n",
    "        # print(str(i) + ': ' + str(len(data)) + ' | ' + str(num_sample_train) + ' | ' + str(num_sample_test))\n",
    "        \n",
    "        for x in data[???] : # 前 80% 資料存進 training list\n",
    "            train_inputs.append(???)\n",
    "            train_labels.append(???)\n",
    "            \n",
    "        for x in data[???] : # 後 20% 資料存進 testing list\n",
    "            test_inputs.append(???)\n",
    "            test_labels.append(???)\n",
    "\n",
    "    train_dataloader = DataLoader(DogDataset(???, ???, ???),\n",
    "                                  batch_size = batch_size, shuffle = True)\n",
    "    test_dataloader = DataLoader(DogDataset(???, ???, ???),\n",
    "                                  batch_size = batch_size, shuffle = False)\n",
    " \n",
    "    return train_dataloader, test_dataloader"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Part 2: 建立 CNN Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 2.2. 自行設計一個新的 CNN model\n",
    "\n",
    "class BuildModel(nn.Module):\n",
    "\n",
    "    def __init__(self):\n",
    "        \n",
    "        super(BuildModel, self).__init__()\n",
    "        \n",
    "        # ----------------------------------------------\n",
    "        ??? # 初始化模型的 layer (input size: 3 * 224 * 224)\n",
    "        # ----------------------------------------------\n",
    "               \n",
    "    def forward(self, x):\n",
    "        \n",
    "        # ----------------------------------------------\n",
    "        ??? # Forward (最後輸出 20 個類別的機率值)\n",
    "        # ----------------------------------------------\n",
    "        return ???"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Setting\n",
    "\n",
    "依據需求調整參數"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "batch_size = 32\n",
    "lr = 1e-3\n",
    "epochs = 20\n",
    "\n",
    "data_dir = 'stanford_dog'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 2.3. \n",
    "\n",
    "train_dataloader, test_dataloader = split_Train_Val_Data(data_dir)\n",
    "\n",
    "C = ???.to(device) # 使用內建的 model 或是自行設計的 model\n",
    "optimizer_C = optim.???(C.parameters(), lr = lr) # 選擇你想用的 optimizer\n",
    "\n",
    "??? # 利用 torchsummary 的 summary package 印出模型資訊，input size: (3 * 224 * 224)\n",
    "\n",
    "# Loss function\n",
    "criteron = ??? # 選擇想用的 loss function"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "loss_epoch_C = []\n",
    "train_acc, test_acc = [], []\n",
    "best_acc, best_auc = 0.0, 0.0"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3. 實作模型訓練和測試模型效能"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "if __name__ == '__main__':    \n",
    "    \n",
    "    for epoch in range(epochs):\n",
    "    \n",
    "        iter = 0\n",
    "        correct_train, total_train = 0, 0\n",
    "        correct_test, total_test = 0, 0\n",
    "        train_loss_C = 0.0\n",
    "\n",
    "        C.??? # 設定 train 或 eval\n",
    "      \n",
    "        print('epoch: ' + str(epoch + 1) + ' / ' + str(epochs))  \n",
    "        \n",
    "        # ---------------------------\n",
    "        # Training Stage\n",
    "        # ---------------------------\n",
    "        \n",
    "        for i, (x, label) in enumerate(train_dataloader) :\n",
    "                     \n",
    "            x, label = x.to(device), label.to(device)\n",
    "                        \n",
    "            ??? # 清空梯度\n",
    "            \n",
    "            ??? # 將訓練資料輸入至模型進行訓練\n",
    "            ??? # 計算 loss\n",
    "            \n",
    "            ??? # 將 loss 反向傳播\n",
    "            ??? # 更新權重\n",
    "            \n",
    "            # 計算訓練資料的準確度 (correct_train / total_train)\n",
    "            _, predicted = ???\n",
    "            total_train += ???\n",
    "            correct_train += ???\n",
    "\n",
    "            train_loss_C += loss.item()\n",
    "            iter += 1\n",
    "                    \n",
    "        print('Training epoch: %d / loss_C: %.3f | acc: %.3f' % \\\n",
    "              (epoch + 1, train_loss_C / iter, correct_train / total_train))\n",
    "\n",
    "        \n",
    "        # --------------------------\n",
    "        # Testing Stage\n",
    "        # --------------------------\n",
    "        \n",
    "        C.??? # 設定 train 或 eval\n",
    "          \n",
    "        for i, (x, label) in enumerate(test_dataloader) :\n",
    "          \n",
    "            with ???: # 測試階段不需要求梯度\n",
    "                x, label = x.to(device), label.to(device)\n",
    "                \n",
    "                ??? # 將測試資料輸入至模型進行測試\n",
    "                ??? # 計算測試資料的準確度\n",
    "                _, predicted = ???\n",
    "                total_test += ???\n",
    "                correct_test += ???\n",
    "        \n",
    "        print('Testing acc: %.3f' % (correct_test / total_test))\n",
    "                                     \n",
    "        train_acc.append(100 * ???) # training accuracy\n",
    "        test_acc.append(100 * ???)  # testing accuracy\n",
    "        loss_epoch_C.append(???) # loss \n",
    "    "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4. 將每一個 epoch 的 Loss 以及 Training / Testing accuracy 紀錄下來並繪製成圖。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure()\n",
    "\n",
    "??? # plot your loss\n",
    "\n",
    "plt.title('Training Loss')\n",
    "plt.ylabel('loss'), plt.xlabel('epoch')\n",
    "plt.legend(['loss_C'], loc = 'upper left')\n",
    "plt.show()\n",
    "\n",
    "plt.figure()\n",
    "\n",
    "??? # plot your training accuracy\n",
    "??? # plot your testing accuracy\n",
    "\n",
    "plt.title('Training acc')\n",
    "plt.ylabel('acc (%)'), plt.xlabel('epoch')\n",
    "plt.legend(['training acc', 'testing acc'], loc = 'upper left')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
