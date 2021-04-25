# 1.1. 填入 ??? 的部份
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
import numpy as np
class DogDataset(Dataset):
    
    def __init__(self, filenames, labels, transform):
        
        self.filenames = filenames # 資料集的所有檔名
        self.labels = labels # 影像的標籤
        self.transform = transform # 影像的轉換方式
 
    def __len__(self):
        
        return len(self.filenames) # return DataSet 長度
 
    def __getitem__(self, idx):
        
        image = Image.open(self.filenames[idx]).convert('RGB')
        image = self.transform(image)
        label = np.array(self.labels[idx])
                
        return image, label # return 模型訓練所需的資訊