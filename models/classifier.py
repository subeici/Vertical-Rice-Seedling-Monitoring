import torch
import torch.nn as nn

from engine.backbone.hgnetv2 import HGNetv2 

class MultiTaskLeafClassifier(nn.Module):
    def __init__(self, pretrained_path=None, hidden_dim=2048):
        super().__init__()
        
        self.backbone = HGNetv2(name='B4', return_idx=[3], use_lab=True, pretrained=False) 
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 🌟 【招式一：失忆药】随机丢弃 50% 的神经元特征
        self.dropout = nn.Dropout(p=0.65)
        
        self.head_age = nn.Linear(hidden_dim, 3)
        self.head_yellow = nn.Linear(hidden_dim, 2)   
        self.head_disease = nn.Linear(hidden_dim, 2)  

        if pretrained_path:
            self.load_backbone_weights(pretrained_path)

    def forward(self, x):
        features = self.backbone(x)
        
        if isinstance(features, list) or isinstance(features, tuple):
            last_feature = features[-1] 
        else:
            last_feature = features
            
        pooled = self.pool(last_feature).flatten(1)
        
        # 🌟 在把特征发给三个分类头之前，先强行喂一颗失忆药
        pooled = self.dropout(pooled)
        
        out_age = self.head_age(pooled)
        out_yellow = self.head_yellow(pooled)
        out_disease = self.head_disease(pooled)
        
        return out_age, out_yellow, out_disease

    def load_backbone_weights(self, weight_path):
        """精准剥离官方权重并安全加载 (带字符串外科手术修复)"""
        print(f"🔄 正在尝试加载预训练权重: {weight_path}")
        try:
            checkpoint = torch.load(weight_path, map_location='cpu')
            if 'ema' in checkpoint:
                state_dict = checkpoint['ema']['module']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
                
            backbone_weights = {}
            for k, v in state_dict.items():
                if 'backbone' in k:
                    new_key = k.replace('backbone.', '')
                    new_key = new_key.replace('.conv.conv.', '.conv.')
                    new_key = new_key.replace('.conv.bn.', '.bn.')
                    backbone_weights[new_key] = v
                    
            if not backbone_weights:
                print("⚠️ 警告: 权重文件中没找到包含 'backbone' 的参数！")
                return
                
            self.backbone.load_state_dict(backbone_weights, strict=False)
            print("✅ 预训练权重加载成功！发动机已被成功点亮！")
        except Exception as e:
            print(f"❌ 加载权重失败: {e}")

if __name__ == '__main__':
    dummy_input = torch.randn(4, 3, 256, 256)
    test_weight_path = './weights/RTv4-L-hgnet.pth' # 修改为你现有的初始权重
    import os
    if not os.path.exists(test_weight_path):
        test_weight_path = None
        
    model = MultiTaskLeafClassifier(pretrained_path=test_weight_path, hidden_dim=2048)
    out_age, out_yellow, out_disease = model(dummy_input)
    
    print("\n✅ --- 网络底盘测试完美通过！---")
    print(f"Age 头输出维度: {out_age.shape} (预期 [4, 2])")