import torch.nn as nn
import torch

class IMCLNet(nn.Module):
    def __init__(self, class_num):
        super(IMCLNet, self).__init__()

        self.class_num = class_num

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # 2
            
            CoordAtt(ch_in=128, ch_out=128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # dw
            nn.Conv2d(128, 128, 3, 1, 1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Hardswish(),

            # pw
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.Hardswish(),

            CoordAtt(ch_in=128, ch_out=128),    # 12
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # 14

            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 1)),        
            CoordAtt(ch_in=64, ch_out=128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # 18

            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 2, 2)),  
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1), 
            nn.ReLU(inplace=True),  # 21
        )

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=448, out_channels=self.class_num, kernel_size=1, stride=1),
        )

    def forward(self, x):

        keep_features = list()

        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)

            if i in [0, 12, 17, 21]:     # IMCLNet                  in_channels=448
                keep_features.append(x)

        global_context = list()
        
        for i, f in enumerate(keep_features):

            f = nn.AdaptiveMaxPool2d(4)(f)
            global_context.append(f)

        x = torch.cat(global_context, 1)

        x = self.feature(x)

        logits = torch.mean(x, dim=[2, 3])

        return logits

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, ch_in, ch_out, reduction=8):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, ch_in // reduction)

        self.conv1 = nn.Conv2d(ch_in, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, ch_in, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, ch_in, kernel_size=1, stride=1, padding=0)
        
        self.conv2 = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        
        _,_,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        out = x * a_w * a_h
        
        out = self.conv2(out)

        return out

class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )
    def forward(self, x):
        return self.block(x)

