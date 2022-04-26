import torch.nn as nn

class SE_Block(nn.Module):
    ##https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c * r, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c * r, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        print(x.shape)
        y = self.squeeze(x).view(bs, c)
        print(y.shape)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)
