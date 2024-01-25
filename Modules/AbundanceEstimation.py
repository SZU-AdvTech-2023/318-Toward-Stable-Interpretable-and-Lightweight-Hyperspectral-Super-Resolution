import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from NetworkBaseModule.blocks import NetBlock


class ConvLayerAbundance(nn.Module):
    def __init__(self,
                 ConvLayerAbundanceParam: dict,
                 bands: int,
                 scale_factor: int):
        super(ConvLayerAbundance, self).__init__()
        self.bands = bands
        self.scale_factor = scale_factor

        self.Encoder_0 = NetBlock(
            mode=ConvLayerAbundanceParam["Encoder_0"]["mode"],
            depth=ConvLayerAbundanceParam["Encoder_0"]["depth"],
            n_channels=ConvLayerAbundanceParam["Encoder_0"]["bands"],
            type_layers=ConvLayerAbundanceParam["Encoder_0"]["layers"],
            param_layers=ConvLayerAbundanceParam["Encoder_0"]["layers_param"]
        )
        self.Encoder_1 = NetBlock(
            mode=ConvLayerAbundanceParam["Encoder_1"]["mode"],
            depth=ConvLayerAbundanceParam["Encoder_1"]["depth"],
            n_channels=ConvLayerAbundanceParam["Encoder_1"]["bands"],
            type_layers=ConvLayerAbundanceParam["Encoder_1"]["layers"],
            param_layers=ConvLayerAbundanceParam["Encoder_1"]["layers_param"]
        )
        self.Encoder_2 = NetBlock(
            mode=ConvLayerAbundanceParam["Encoder_2"]["mode"],
            depth=ConvLayerAbundanceParam["Encoder_2"]["depth"],
            n_channels=ConvLayerAbundanceParam["Encoder_2"]["bands"],
            type_layers=ConvLayerAbundanceParam["Encoder_2"]["layers"],
            param_layers=ConvLayerAbundanceParam["Encoder_2"]["layers_param"]
        )
        self.Encoder_3 = NetBlock(
            mode=ConvLayerAbundanceParam["Encoder_3"]["mode"],
            depth=ConvLayerAbundanceParam["Encoder_3"]["depth"],
            n_channels=ConvLayerAbundanceParam["Encoder_3"]["bands"],
            type_layers=ConvLayerAbundanceParam["Encoder_3"]["layers"],
            param_layers=ConvLayerAbundanceParam["Encoder_3"]["layers_param"]
        )
        self.Encoder_4 = NetBlock(
            mode=ConvLayerAbundanceParam["Encoder_4"]["mode"],
            depth=ConvLayerAbundanceParam["Encoder_4"]["depth"],
            n_channels=ConvLayerAbundanceParam["Encoder_4"]["bands"],
            type_layers=ConvLayerAbundanceParam["Encoder_4"]["layers"],
            param_layers=ConvLayerAbundanceParam["Encoder_4"]["layers_param"]
        )

    def forward(self, X: Tensor, Y: Tensor, scale_factor):

        Temp = Y.permute(0, 3, 1, 2)
        X = X.permute(0, 3, 1, 2)

        ratio = Temp.shape[2] / X.shape[2]

        X_up = F.interpolate(X, scale_factor=ratio, mode='bilinear')
        # Temp = torch.cat((Temp, X_up), 1)

        # conv_layer = nn.Conv2d(in_channels=34, out_channels=3, kernel_size=1).to("cuda")

        # Temp = conv_layer(Temp)
        

        batchsize, height, width, bands_msi = Y.shape
        height_hsi = X.shape[2]

        A_0 = self.Encoder_0(Temp)
        A_1 = self.Encoder_1(torch.cat([A_0, Y.permute(0, 3, 1, 2)], dim=1))
        A_2 = self.Encoder_2(torch.cat([A_1, Y.permute(0, 3, 1, 2)], dim=1))
        A_3 = self.Encoder_3(torch.cat([A_2, Y.permute(0, 3, 1, 2)], dim=1))
        A = self.Encoder_4(torch.cat([A_3, Y.permute(0, 3, 1, 2)], dim=1))

        # conv2d_layer1 = nn.Conv2d(15, 11, kernel_size=1).to('cuda')
        # conv2d_layer2 = nn.Conv2d(34, 19, kernel_size=1).to('cuda')
        # conv2d_layer3 = nn.Conv2d(61, 27, kernel_size=1).to('cuda')


        # A_0 = self.Encoder_0(Temp)
        # A_1 = self.Encoder_1(torch.cat([A_0, Y.permute(0, 3, 1, 2)], dim=1))
        # A_1 = torch.cat([A_0, A_1, Y.permute(0, 3, 1, 2)], dim=1)
        # # A_1 = conv2d_layer1(A_1_1)
        # A_2 = self.Encoder_2(A_1)
        # A_2 = torch.cat([A_0, A_1, A_2, Y.permute(0, 3, 1, 2)], dim=1)
        # # A_2 = conv2d_layer2(A_2_1)
        # A_3 = self.Encoder_3(A_2)
        # A_3 = torch.cat([A_0, A_1, A_2, A_3, Y.permute(0, 3, 1, 2)], dim=1)
        # # A_3 = conv2d_layer3(A_3_1)
        # A = self.Encoder_4(A_3)

        

        

        return A
