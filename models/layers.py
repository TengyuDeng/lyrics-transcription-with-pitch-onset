import sys
sys.path.append("..")
from utils import downsample_length, get_padding
import os
from torch import nn

class CNNLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.2):

        super(CNNLayer, self).__init__()

        self.norm = nn.InstanceNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=get_padding(kernel_size))

    def forward(self, x):

        return self.conv(self.dropout(self.relu(self.norm(x))))


class ResCNNLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0.2):

        super(ResCNNLayer, self).__init__()

        self.norm = nn.InstanceNorm2d(in_channels)
        if in_channels == out_channels and stride == 1:
            self.conv1 = nn.Identity()
        else:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=get_padding(kernel_size))

    def forward(self, x):

        x = self.conv1(self.norm(x))
        return self.conv2(self.dropout(self.relu(x))) + x

        
class RNNLayer(nn.Module):

    def __init__(self, input_size, hidden_size, dropout=0.2, normalize=False):

        super(RNNLayer, self).__init__()

        self.norm = nn.LayerNorm(input_size) if normalize else nn.Identity()
        self.bilstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x, (h_n, c_n) = self.bilstm(self.norm(x))
        return self.dropout(x)

class TCNLayer(nn.Module):

    def __init__(self, ):
        pass