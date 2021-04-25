import torch.nn as nn

class BuildModel(nn.Module):

    def __init__(self):
        
        super(BuildModel, self).__init__()
        
        self.VGG16_form = [64,64,"M",128,128,"M",256,256,256,"M",512,512,512,"M",512,512,512,"M"]
        self.in_channels = 3
        self.img_size = 224
        self.num_classes = 20
        self.conv_layers = self.build_conv_layers(self.VGG16_form)
        self.fully_connected_layers = nn.Sequential(
        nn.Linear(int(((self.img_size/32)**2)*512), 1024), #conv最後從(7 X 7 X 512)flatten為一維
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(p= 0.5),
        nn.Linear(1024,1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(p= 0.5),
        nn.Linear(1024, self.num_classes))
        
        
        # ----------------------------------------------
        # 初始化模型的 layer (input size: 3 * 224 * 224)
        # ----------------------------------------------
               
    def forward(self, x):
        
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fully_connected_layers(x)
        # ----------------------------------------------
        # Forward (最後輸出 20 個類別的機率值)
        # ----------------------------------------------
        return x
    
    def build_conv_layers(self, VGG_architecture): #建立VGG16架構
        layers = []
        in_channels = self.in_channels

        for layer_out in VGG_architecture: #針對vgg16的結構建立layers(list)
            if type(layer_out) == int:

                out_channels = layer_out

                layers += [nn.Conv2d(in_channels= in_channels, out_channels= out_channels,
                kernel_size=(3,3), stride=(1,1), padding=(1,1)),nn.BatchNorm2d(out_channels) ,nn.ReLU()]

                in_channels = out_channels #更新下一個layers的in_channel
            elif layer_out == 'M': #沒有參數
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride= (2,2))]
        return nn.Sequential(*layers)