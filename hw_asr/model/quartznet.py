from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel


class Conv_Bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2, dilation=dilation),
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, R, in_channels, out_channels, kernel_size, dropout=0.2):
        super().__init__()

        self.net = [Conv_Bn(in_channels, out_channels, kernel_size)]
        for i in range(R - 2):
            self.net.append(nn.ReLU())
            self.net.append(Conv_Bn(out_channels, out_channels, kernel_size))

        self.net = nn.Sequential(*self.net)

        self.second_net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(nn.ReLU(self.net(x) + self.second_net(x)))


class QuartzNet(BaseModel):
    def __init__(self, n_features, n_class, *args, **kwargs):
        super(QuartzNet, self).__init__(n_features, n_class, *args, **kwargs)

        self.c_1 = Conv_Bn(n_features, 256, 33, stride=2)
        self.b_1 = Block(5, 256, 256, 33)
        self.b_2 = Block(5, 256, 256, 39)
        self.b_3 = Block(5, 256, 512, 51)
        self.b_4 = Block(5, 512, 512, 63)
        self.b_5 = Block(5, 512, 512, 75)
        self.c_2 = Conv_Bn(512, 512, 75)
        self.c_3 = Conv_Bn(512, 1024, 1)
        self.c_4 = nn.Conv1d(1024, n_class, dilation=2, bias=False)

    def forward(self, x):
        x = x.transpose(-2, -1)
        x = nn.ReLU(self.c_1(x))
        x = self.b_1(x)
        x = self.b_2(x)
        x = self.b_3(x)
        x = self.b_4(x)
        x = self.b_5(x)
        x = nn.ReLU(self.c_2(x))
        x = nn.ReLU(self.c_3(x))
        x = self.c_4(x)
        return {"logits": x.transpose(-2, -1)}

    def transform_input_lengths(self, input_lengths):
        return input_lengths // 2
