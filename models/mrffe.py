import torch
import torch.nn as nn
import torch.nn.functional as F

class MRFFE(nn.Module):
    """
    Multi-Resolution Feature Fusion and Enhancement (MRFFE) Module
    多分支特征融合增强模块，包含:
    - 三个并行特征提取分支(M1, M2, M3)
    - 特征拼接融合
    - 残差连接
    """
    def __init__(self, in_channels, out_channels=None, reduction=4):
        """
        Args:
            in_channels: 输入特征图通道数
            out_channels: 输出特征图通道数，默认与输入相同
            reduction: 通道缩减比例，用于控制中间特征通道数
        """
        super(MRFFE, self).__init__()
        self.out_channels = out_channels if out_channels else in_channels
        mid_channels = in_channels // reduction
        
        # 1x1卷积用于输入特征调整 (S模块)
        self.conv1x1_in = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn_in = nn.BatchNorm2d(mid_channels)
        self.relu_in = nn.ReLU(inplace=False)
        
        # 分支M1: 1x3 Conv -> 3x1 Conv -> 3x3 Conv
        self.m1_conv1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(1, 3), padding=(0, 1), bias=False)                                                                                                                               # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        self.m1_bn1 = nn.BatchNorm2d(mid_channels)
        self.m1_relu1 = nn.ReLU(inplace=False)
        self.m1_conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(3, 1), padding=(1, 0), bias=False)                                                                                                                               # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        self.m1_bn2 = nn.BatchNorm2d(mid_channels)
        self.m1_relu2 = nn.ReLU(inplace=False)
        self.m1_conv3 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.m1_bn3 = nn.BatchNorm2d(mid_channels)
        self.m1_relu3 = nn.ReLU(inplace=False)
        
        # 分支M2: 3x1 Conv -> 1x3 Conv -> 3x3 Conv
        self.m2_conv1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(3, 1), padding=(1, 0), bias=False)                                                                                                                               # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        self.m2_bn1 = nn.BatchNorm2d(mid_channels)
        self.m2_relu1 = nn.ReLU(inplace=False)
        self.m2_conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(1, 3), padding=(0, 1), bias=False)                                                                                                                               # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        self.m2_bn2 = nn.BatchNorm2d(mid_channels)
        self.m2_relu2 = nn.ReLU(inplace=False)
        self.m2_conv3 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.m2_bn3 = nn.BatchNorm2d(mid_channels)
        self.m2_relu3 = nn.ReLU(inplace=False)
        
        # 分支M3: 3x3 Conv (R=5) -> 3x3 Conv -> 3x3 Conv
        # R=5膨胀卷积(dilation=2)，此时感受野约为5
        self.m3_conv1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=2, dilation=2, bias=False)
        self.m3_bn1 = nn.BatchNorm2d(mid_channels)
        self.m3_relu1 = nn.ReLU(inplace=False)
        self.m3_conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.m3_bn2 = nn.BatchNorm2d(mid_channels)
        self.m3_relu2 = nn.ReLU(inplace=False)
        self.m3_conv3 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.m3_bn3 = nn.BatchNorm2d(mid_channels)
        self.m3_relu3 = nn.ReLU(inplace=False)
        
        # 融合后处理 (M模块)
        self.conv1x1_out = nn.Conv2d(mid_channels * 3, self.out_channels, kernel_size=1, bias=False)
        self.bn_out = nn.BatchNorm2d(self.out_channels)
        self.relu_out = nn.ReLU(inplace=False)
        
        # 残差连接调整 (如果输入输出通道不同)
        if in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, bias=False)
            self.bn_shortcut = nn.BatchNorm2d(self.out_channels)
        else:
            self.shortcut = nn.Identity()
            self.bn_shortcut = nn.Identity()

    def forward(self, x):
        """前向传播过程"""
        # 保存原始输入用于残差连接
        residual = x
        
        # 输入特征调整 (S模块)
        x = self.conv1x1_in(x)
        x = self.bn_in(x)
        x = self.relu_in(x)
        
        # 分支M1前向传播
        m1 = self.m1_conv1(x)
        m1 = self.m1_bn1(m1)
        m1 = self.m1_relu1(m1)
        m1 = self.m1_conv2(m1)
        m1 = self.m1_bn2(m1)
        m1 = self.m1_relu2(m1)
        m1 = self.m1_conv3(m1)
        m1 = self.m1_bn3(m1)
        m1 = self.m1_relu3(m1)
        
        # 分支M2前向传播
        m2 = self.m2_conv1(x)
        m2 = self.m2_bn1(m2)
        m2 = self.m2_relu1(m2)
        m2 = self.m2_conv2(m2)
        m2 = self.m2_bn2(m2)
        m2 = self.m2_relu2(m2)
        m2 = self.m2_conv3(m2)
        m2 = self.m2_bn3(m2)
        m2 = self.m2_relu3(m2)
        
        # 分支M3前向传播
        m3 = self.m3_conv1(x)
        m3 = self.m3_bn1(m3)
        m3 = self.m3_relu1(m3)
        m3 = self.m3_conv2(m3)                                                                                                                               # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        m3 = self.m3_bn2(m3)
        m3 = self.m3_relu2(m3)
        m3 = self.m3_conv3(m3)
        m3 = self.m3_bn3(m3)
        m3 = self.m3_relu3(m3)                                                                                                                               # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        
        # 特征拼接融合 (C模块)
        fused = torch.cat([m1, m2, m3], dim=1)                                                                                                                               # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        
        # 融合后处理 (M模块)
        out = self.conv1x1_out(fused)
        out = self.bn_out(out)
        out = self.relu_out(out)
        
        # 残差连接
        residual = self.shortcut(residual)
        residual = self.bn_shortcut(residual)
        out = out + residual
        
        return out

# 使用示例
if __name__ == "__main__":
    # 创建一个随机输入特征图 
    input_tensor = torch.randn(1, 32, 256, 256)  # (batch_size, channels, height, width)                                                                                                                               # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
    
    # 初始化MRFFE模块
    mrffe = MRFFE(in_channels=32)
    print(mrffe)
    
    # 前向传播
    output_tensor = mrffe(input_tensor)
    
    # 打印输入输出形状
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")
