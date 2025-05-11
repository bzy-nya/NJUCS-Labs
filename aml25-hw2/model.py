import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class NekoNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # 28x28x1 -> 28x28x16
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)                          # 28x28x16 -> 14x14x16
        self.dropout1 = nn.Dropout2d(0.25)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # 14x14x16 -> 14x14x32
        self.bn2 = nn.BatchNorm2d(32)
        self.se_block = SEBlock(32) 

        self.skip_conv = nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.skip_bn = nn.BatchNorm2d(32)

        self.pool2 = nn.MaxPool2d(2, 2)                          # 14x14x32 -> 7x7x32
        self.dropout2 = nn.Dropout2d(0.25)

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)

        identity = x 

        out = self.conv2(x)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.se_block(out)

        skip = self.skip_conv(identity)
        skip = self.skip_bn(skip)
        
        x = out + skip
        x = F.relu(x)

        x = self.pool2(x)
        x = self.dropout2(x)

        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x