import torch
import torch.nn as nn
import brevitas.nn as qnn


class LeNet5Quantized(nn.Module):

    def __init__(
        self,
        num_classes:      int = 10,
        weight_bit_width: int = 4,
        act_bit_width:    int = 4,
        inp_bit_width:    int = 8,
        in_channels:      int = 1,
    ):
        super().__init__()

        self.quant_inp = qnn.QuantIdentity(
            bit_width=inp_bit_width,
            return_quant_tensor=True,
        )

        self.conv_0 = qnn.QuantConv2d(
            in_channels=in_channels, out_channels=6,
            kernel_size=5, stride=1, padding=0, bias=False,
            weight_bit_width=weight_bit_width,
            return_quant_tensor=True,
        )
        self.act_0  = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)
        self.pool_0 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_1 = qnn.QuantConv2d(
            in_channels=6, out_channels=16,
            kernel_size=5, stride=1, padding=0, bias=False,
            weight_bit_width=weight_bit_width,
            return_quant_tensor=True,
        )
        self.act_1  = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_2 = qnn.QuantConv2d(
            in_channels=16, out_channels=120,
            kernel_size=5, stride=1, padding=0, bias=False,
            weight_bit_width=weight_bit_width,
            return_quant_tensor=True,
        )
        self.act_2  = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)

        self.flatten  = nn.Flatten()
        self.fc_0     = qnn.QuantLinear(
            in_features=120, out_features=84, bias=False,
            weight_bit_width=weight_bit_width,
            return_quant_tensor=True,
        )
        self.fc_act_0 = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)

        self.fc_1 = qnn.QuantLinear(
            in_features=84, out_features=num_classes, bias=False,
            weight_bit_width=weight_bit_width,
            return_quant_tensor=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant_inp(x)
        x = self.conv_0(x)
        x = self.act_0(x)
        x = self.pool_0(x)
        x = self.conv_1(x)
        x = self.act_1(x)
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = self.act_2(x)
        x = self.flatten(x)
        x = self.fc_0(x)
        x = self.fc_act_0(x)
        x = self.fc_1(x)
        return x