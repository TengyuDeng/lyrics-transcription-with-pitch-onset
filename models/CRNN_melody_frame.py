import sys
sys.path.append("..")
from utils import downsample_length, get_padding

from torch import nn
from .layers import *


class CRNN_melody_frame(nn.Module):

    def __init__(
        self,
        num_classes_pitch,
        input_features=80,
        input_channels=1,
        num_convs=6,
        conv_channels=[64, 32, 32, 32, 32, 1],
        kernel_sizes=[5, 5, 3, 3, 3, 1],
        dropout=0.,
        lstm_channels=512,
        lstm_norm=True,
        **args
        ):

        super(CRNN_melody_frame, self).__init__()

        if len(conv_channels) != num_convs:
            raise ValueError(f"Expect conv_channels to have {num_convs} elements but got {len(conv_channels)}!")
        if len(kernel_sizes) != num_convs:
            raise ValueError(f"Expect kernel_sizes to have {num_convs} elements but got {len(kernel_sizes)}!")
        self.num_classes_pitch = num_classes_pitch
        self.input_features = input_features
        
        self.input_channels = input_channels
        self.conv_channels = conv_channels
        self.kernel_sizes = kernel_sizes
        
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=conv_channels[0], 
                kernel_size=kernel_sizes[0], 
                padding=get_padding(kernel_sizes[0]),
                ),
            *[
            ResCNNLayer(
                in_channels=conv_channels[i - 1], 
                out_channels=conv_channels[i], 
                kernel_size=kernel_sizes[i],
                dropout=dropout,
                )
            for i in range(1, num_convs)
            ],
            )

        self.rnn_pitch = RNNLayer(
            input_size=conv_channels[-1] * input_features, 
            hidden_size=num_classes_pitch + 1, 
            dropout=dropout,
            normalize=lstm_norm,
        )

    def forward(self, x):
            
        if x.shape[-2] != self.input_features:
            raise ValueError(f"Number of input features not match! expected{self.input_features} but got {x.shape[-2]}")
        
        if self.input_channels == 1:
            x = x.unsqueeze(-3)

        # x: (batch_size, channel=1, feature_num, length)

        x = self.cnn(x)
        # x: (batch_size, channel=conv_channels, new_feature_num, length)

#         tatum_frames = tatum_frames.unsqueeze(-2)
        # x: (batch_size, channel * feature_num, length)
        old_shape = x.shape
        x_pitch = x.reshape(old_shape[0], old_shape[1] * old_shape[2], old_shape[3])
        # x: (batch_size, channel * new_feature_num, new_length)

        x_pitch = x_pitch.permute(2, 0, 1)
        # x_pitch: (num_tatums, batch_size, feature_num)

        x_pitch = self.rnn_pitch(x_pitch)
        x_pitch = x_pitch[:,:,:self.num_classes_pitch + 1] + x_pitch[:,:,self.num_classes_pitch + 1:]
        # x_pitch: (num_tatums, batch_size, num_classes + 1) -> (batch_size, num_classes + 1, num_tatums)
        output_pitch = x_pitch.permute(1, 2, 0)
      
        return output_pitch

        # output_pitch: (batch_size, num_classes + 1, num_tatums)
        # output_lyrics: (new_length, batch_size, num_classes)
