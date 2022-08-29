import sys
sys.path.append("..")
from utils import downsample_length, get_padding

import torch
from torch import nn
from .layers import *


class CRNN_lyrics_onset(nn.Module):

    def __init__(
        self,
        num_classes_lyrics,
        input_features=80,
        input_channels=1,
        num_convs=6,
        conv_channels=[64, 32, 32, 32, 32, 16],
        kernel_sizes=[5, 5, 3, 3, 3, 3],
        dropout=0.,
        num_lstms=3,
        lstm_channels=512,
        down_sample=2,
        **args
        ):

        super(CRNN_lyrics_onset, self).__init__()

        if len(conv_channels) != num_convs:
            raise ValueError(f"Expect conv_channels to have {num_convs} elements but got {len(conv_channels)}!")
        if len(kernel_sizes) != num_convs:
            raise ValueError(f"Expect kernel_sizes to have {num_convs} elements but got {len(kernel_sizes)}!")
        self.num_classes_lyrics = num_classes_lyrics
        self.input_features = input_features
        
        self.input_channels = input_channels
        self.conv_channels = conv_channels
        self.kernel_sizes = kernel_sizes
        self.down_sample = down_sample
        
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
        
        self.pooling_lyrics = nn.MaxPool2d(
            kernel_size=3, 
            stride=self.down_sample, 
            padding=get_padding(3),
        )

        self.pooling_onsets = nn.MaxPool1d(
            kernel_size=3,
            stride=self.down_sample, 
            padding=get_padding(3),
        )
        # self.downsampling_lyrics = ResCNNLayer(
        #     in_channels=conv_channels[-1],
        #     out_channels=conv_channels[-1],
        #     kernel_size=kernel_sizes[-1],
        #     stride=self.down_sample,
        #     dropout=dropout,
        # )

        self.rnn_lyrics = nn.Sequential(
            *[
            RNNLayer(
            input_size=lstm_channels * 2 if i > 0 else conv_channels[-1] * downsample_length(input_features, self.down_sample) + 1, 
            hidden_size=lstm_channels,
            dropout=dropout,
            normalize=True if i > 0 else False,
            )
            for i in range(num_lstms)
            ]
            )
        self.dense_lyrics = nn.Sequential(
            nn.Linear(in_features=lstm_channels * 2, out_features=lstm_channels),
            nn.ReLU(),
            nn.Linear(in_features=lstm_channels, out_features=num_classes_lyrics),
            )


    def forward(self, x, onsets):
            
        if x.shape[-2] != self.input_features:
            raise ValueError(f"Number of input features not match! expected{self.input_features} but got {x.shape[-2]}")
        # if tatum_frames.shape[-1] != self.num_tatums + 1:
        #     raise ValueError(f"Length of tatum_frames not match! expected{self.num_tatums + 1} but got {tatum_frames.shape[-1]}")
        
        if self.input_channels == 1:
            x = x.unsqueeze(-3)

        # x: (batch_size, channel=1, feature_num, length)
        # tatum_frames: (batch_size, channel=1, num_tatums + 1)

        x = self.cnn(x)
        # x: (batch_size, channel=conv_channels, new_feature_num, new_length)
        
        x_lyrics = self.pooling_lyrics(x)
        onsets = self.pooling_onsets(onsets.unsqueeze(-2))

        # x_lyrics = self.downsampling_lyrics(x)
        # x_lyrics: (batch_size, channel, new_feature_num, new_length)
        old_shape = x_lyrics.shape
        x_lyrics = x_lyrics.reshape(old_shape[0], old_shape[1] * old_shape[2], old_shape[3])
        # x: (batch_size, channel * new_feature_num, new_length)
        x_lyrics = torch.cat([x_lyrics, onsets], dim=1)
        x_lyrics = x_lyrics.permute(2, 0, 1)
        # x_lyrics: (new_length, batch_size, feature_num)

        x_lyrics = self.rnn_lyrics(x_lyrics)
        output_lyrics = self.dense_lyrics(x_lyrics)
        # x_lyrics: (new_length, batch_size, num_classes)

        return output_lyrics
        # output_pitch: (batch_size, num_classes + 1, num_tatums)
        # output_lyrics: (new_length, batch_size, num_classes)
