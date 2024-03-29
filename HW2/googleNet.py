import torch 
import torch.nn as nn 

# input size doesnt change with inception block but filter size does
class Inception_block(nn.Module):
	def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
		super(Inception_block, self).__init__()

		self.branch1 = conv_block(in_channels, out_1x1, kernel_size= 1)

		self.branch2 = nn.Sequential(
			conv_block(in_channels, red_3x3, kernel_size= 1),
			conv_block(red_3x3, out_3x3, kernel_size= 3, stride= 1, padding= 1))

		self.branch3 = nn.Sequential(
			conv_block(in_channels, red_5x5, kernel_size= 1),
			conv_block(red_5x5, out_5x5, kernel_size= 5, stride= 1, padding= 2))

		self.branch4 = nn.Sequential(
			nn.MaxPool2d(kernel_size= 3, stride= 1, padding= 1),
			conv_block(in_channels, out_1x1pool, kernel_size= 1))

		def forward(self, x):
			# input size : (N, filters, 28, 28)
			return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim= 1)
			# cat dim default 0 --> concattenate over rows 

# Inception Block uses conv_block with diff kernels 
class conv_block(nn.Module):
	def __init__(self, in_channels, out_channels, **kwargs):
		super(conv_block, self).__init__()
		self.relu = nn.ReLU()
		self.conv = nn.Conv2d(in_channels., out_channels, **kwargs)
		self.batchnorm = nn.BatchNorm2d(out_channels)

		def forward(self,x):
			return self.relu(self.batchnorm(self.conv(x)))

