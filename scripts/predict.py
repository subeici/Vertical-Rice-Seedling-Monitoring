import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

# 导入你的主厨模型
from model_leaf import MultiTaskLeafClassifier

def predict_single_image(image_path, model_weight_path):
    # 1. 基础设置与标签映射表
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 正在使用设备: {device} 进行推理")
    
    # 这里的映射必须和 dataset_leaf.py 里的生成逻辑完全一致
    map_age = {0: "一叶 (one)", 1: "二叶 (two)", 2:"三叶(three)"}
    map_yellow = {0: "不发黄 (no)", 1: "发黄 (yes)"}
    map_disease = {0: "无病 (no)", 1: "有病 (yes)"}

    # 2. 图像预处理管道 (必须和验证集 Val 保持绝对一致！)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # 3. 加载图像并预处理
    if not os.path.exists(image_path):
        print(f"❌ 找不到图片: {image_path}")
        return
        
    print(f"📸 正在读取图片: {image_path}")
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0) # 增加 Batch 维度，变成 [1, 3, 256, 256]
    input_tensor = input_tensor.to(device)

    # 4. 构建模型并加载你训练好的权重
    # 记得 hidden_dim=2048 对应 B4 发动机
    model = MultiTaskLeafClassifier(pretrained_path=None, hidden_dim=2048)
    
    if not os.path.exists(model_weight_path):
        print(f"❌ 找不到权重文件: {model_weight_path}，请确认训练是否成功保存。")
        return
        
   print(f"🧠 正在加载模型权重: {model_weight_path}")
    # 智能读取：先把文件加载到内存
    checkpoint = torch.load(model_weight_path, map_location=device)
    # 兼容新版包含断点信息的字典
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 兼容老版直接保存权重的格式
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval() # 切记：推理时必须开启 eval 模式，关闭 Dropout 和 BatchNorm 的随机性

    # 5. 执行前向传播并解析结果
    print("-" * 40)
    print("🔍 模型诊断结果：")
    with torch.no_grad(): # 推理时不需要计算梯度，省显存提速
        out_age, out_yellow, out_disease = model(input_tensor)
        
        # 将输出的 Logits 转换为概率 (0~100%)
        prob_age = F.softmax(out_age, dim=1)[0]
        prob_yellow = F.softmax(out_yellow, dim=1)[0]
        prob_disease = F.softmax(out_disease, dim=1)[0]
        
        # 获取得分最高的索引
        pred_age_idx = torch.argmax(prob_age).item()
        pred_yellow_idx = torch.argmax(prob_yellow).item()
        pred_disease_idx = torch.argmax(prob_disease).item()
        
        # 打印详细结果
        print(f"🌱 【叶龄】: {map_age[pred_age_idx]} (确信度: {prob_age[pred_age_idx].item()*100:.2f}%)")
        print(f"🍂 【缺水/发黄】: {map_yellow[pred_yellow_idx]} (确信度: {prob_yellow[pred_yellow_idx].item()*100:.2f}%)")
        print(f"🦠 【病害】: {map_disease[pred_disease_idx]} (确信度: {prob_disease[pred_disease_idx].item()*100:.2f}%)")
        print("-" * 40)

if __name__ == '__main__':
    # ================= 修改区 =================
    # 把这里的路径换成你想测试的任何一张图片的绝对路径或相对路径
    # 例如你可以从验证集里随便挑一张：'./data/Processed_Leaf_Dataset/images/val/val_crop_000001.jpg'
    TEST_IMAGE_PATH = 'ScreenShot_2026-03-08_135618_611.png' 
    
    # 你刚刚训练出来的最强权重的路径
    BEST_WEIGHT_PATH = './weights/checkpoints/best_leaf_model.pth'
    # ==========================================
    
    predict_single_image(TEST_IMAGE_PATH, BEST_WEIGHT_PATH)