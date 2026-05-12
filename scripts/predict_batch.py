import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import csv

# 导入你的主厨模型
from model_leaf import MultiTaskLeafClassifier

def predict_batch_images(folder_path, model_weight_path, output_csv_path='./picture_predict_results.csv'):
    # 1. 基础设置与标签映射表
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 正在使用设备: {device} 进行批量推理")
    
    map_age = {0: "一叶", 1: "二叶"}
    map_yellow = {0: "不发黄", 1: "发黄"}
    map_disease = {0: "无病", 1: "有病"}

    # 2. 预处理管道
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # 3. 提前加载模型 (核心优化：只加载一次！)
    if not os.path.exists(model_weight_path):
        print(f"❌ 找不到权重文件: {model_weight_path}")
        return
        
    print(f"🧠 正在加载模型权重: {model_weight_path}")
    model = MultiTaskLeafClassifier(pretrained_path=None, hidden_dim=2048)
    
    # 智能读取：先把文件加载到内存
    checkpoint = torch.load(model_weight_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval() # 开启推理模式

    # 4. 检查文件夹并获取所有图片
    if not os.path.exists(folder_path):
        print(f"❌ 找不到目标文件夹: {folder_path}")
        return
        
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"⚠️ 文件夹 {folder_path} 中没有找到支持的图片文件！")
        return

    print(f"📁 找到 {len(image_files)} 张图片，开始流水线检测...")
    
    # 5. 准备写入 CSV
    results_list = []
    
    with torch.no_grad():
        for idx, img_name in enumerate(image_files):
            img_path = os.path.join(folder_path, img_name)
            
            # 读图并处理
            try:
                image = Image.open(img_path).convert('RGB')
                input_tensor = transform(image).unsqueeze(0).to(device)
            except Exception as e:
                print(f"⚠️ 读取图片 {img_name} 失败跳过，原因: {e}")
                continue
                
            # 前向传播
            out_age, out_yellow, out_disease = model(input_tensor)
            
            # 算概率
            prob_age = F.softmax(out_age, dim=1)[0]
            prob_yellow = F.softmax(out_yellow, dim=1)[0]
            prob_disease = F.softmax(out_disease, dim=1)[0]
            
            # 取最大值
            pred_age_idx = torch.argmax(prob_age).item()
            pred_yellow_idx = torch.argmax(prob_yellow).item()
            pred_disease_idx = torch.argmax(prob_disease).item()
            
            # 提取文本结果和确信度
            res_age = map_age[pred_age_idx]
            conf_age = f"{prob_age[pred_age_idx].item()*100:.2f}%"
            
            res_yellow = map_yellow[pred_yellow_idx]
            conf_yellow = f"{prob_yellow[pred_yellow_idx].item()*100:.2f}%"
            
            res_disease = map_disease[pred_disease_idx]
            conf_disease = f"{prob_disease[pred_disease_idx].item()*100:.2f}%"
            
            # 存入列表
            results_list.append([
                img_name, res_age, conf_age, res_yellow, conf_yellow, res_disease, conf_disease
            ])
            
            # 终端进度打印
            if (idx + 1) % 10 == 0 or (idx + 1) == len(image_files):
                print(f"🔄 进度: {idx + 1}/{len(image_files)}...")

    # 6. 保存到 CSV
    with open(output_csv_path, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['图片名称', '叶龄预测', '叶龄确信度', '发黄预测', '发黄确信度', '病害预测', '病害确信度'])
        writer.writerows(results_list)
        
    print("-" * 40)
    print(f"🎉 批量检测完成！共处理 {len(results_list)} 张图片。")
    print(f"📊 详细检测报告已保存至: {os.path.abspath(output_csv_path)}")

if __name__ == '__main__':
    # ================= 修改区 =================
    # 你的图片文件夹相对路径
    FOLDER_PATH = './picture' 
    
    # 你的最佳权重路径
    BEST_WEIGHT_PATH = './weights/checkpoints/best_leaf_model.pth'
    
    # 你想把表格保存在哪里，叫什么名字
    CSV_OUTPUT_PATH = './picture_predict_results.csv'
    # ==========================================
    
    predict_batch_images(FOLDER_PATH, BEST_WEIGHT_PATH, CSV_OUTPUT_PATH)