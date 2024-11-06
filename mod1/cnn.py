from torch import nn
import torch.nn.functional as F

KERNEL_SIZE = 5
STRIDE_VALUE = 1
PADDING_VALUE = 2
LAYERS_NEURONS = [120,84,10]

class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()
        # since we have the images in grayscale not rgb
        # we keept the out_channels in 6 to simplify, and i may assume the images dont have that much detail to absorb
        # a kernel size of 5 should be enough for 28x28 images
        # ((28-5+2*2)/1)+1 = 28 so keep the padding in 2 (LeNet)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=KERNEL_SIZE, stride=STRIDE_VALUE, padding=PADDING_VALUE)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=KERNEL_SIZE)
        
        # 120 neurons (may try improve for better results)
        self.fc1 = nn.Linear(16 * KERNEL_SIZE * KERNEL_SIZE, LAYERS_NEURONS[0]) # 400, 120
        self.fc2 = nn.Linear(LAYERS_NEURONS[0], LAYERS_NEURONS[1]) # 120, 84 //LeNet-5 said 84 is good, who am i to disagree 
        self.fc3 = nn.Linear(LAYERS_NEURONS[1], LAYERS_NEURONS[2]) # 84, 10

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2) # 28x28 -> 14x14
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2) # 10x10 -> 5x5 
        
        # Flatten (-1 sets the batch_size automatically)
        x = x.view(-1, 16 * KERNEL_SIZE * KERNEL_SIZE)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Output layer (no activation function, we will be returning the raw scores)
        x = self.fc3(x)
        return x