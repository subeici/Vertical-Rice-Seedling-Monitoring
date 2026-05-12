import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class LeafMultiTaskDataset(Dataset):
    def __init__(self, data_root, split='train', transform=None):
        self.data_root = data_root
        self.split = split
        self.transform = transform

        # 构建路径
        self.img_dir = os.path.join(data_root, split, 'images')
        self.json_path = os.path.join(data_root, split, 'annotations', 'instances.json')

        # 读取 COCO JSON
        with open(self.json_path, 'r', encoding='utf-8') as f:
            coco = json.load(f)

        # 建立 image_id -> 图像信息（文件名、宽、高）的映射
        self.images = {img['id']: img for img in coco['images']}

        # 收集所有 annotations，每个 annotation 视为一个样本
        self.samples = []  # 每个元素为 (img_path, bbox, age_label, yellow_label, disease_label)
        for ann in coco['annotations']:
            img_id = ann['image_id']
            img_info = self.images[img_id]
            img_path = os.path.join(self.img_dir, img_info['file_name'])
            bbox = ann['bbox']  # [x, y, width, height]
            age = ann['age_label']
            yellow = ann['yellow_label']
            disease = ann['disease_label']
            self.samples.append((img_path, bbox, age, yellow, disease))

        print(f"[{split.upper()} Dataset] 加载成功，共找到 {len(self.samples)} 个叶片样本。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, bbox, age, yellow, disease = self.samples[idx]

        # 读取整图
        image = Image.open(img_path).convert('RGB')

        # 根据 bbox 裁剪叶片区域
        x, y, w, h = bbox
        # bbox 是浮点数，需转为整数
        x, y, w, h = int(x), int(y), int(w), int(h)
        # 确保裁剪区域在图像内
        img_w, img_h = image.size
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        crop = image.crop((x, y, x + w, y + h))

        if self.transform:
            crop = self.transform(crop)

        # 转换为 PyTorch 张量
        age_label = torch.tensor(age, dtype=torch.long)
        yellow_label = torch.tensor(yellow, dtype=torch.long)
        disease_label = torch.tensor(disease, dtype=torch.long)

        return crop, {
            'age': age_label,
            'yellow': yellow_label,
            'disease': disease_label
        }


def get_transforms(split='train', img_size=256):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=45),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize
        ])


# ==========================================
# 独立测试模块 (在终端直接运行此文件时触发)
# ==========================================
if __name__ == '__main__':
    # 因为你的代码和 data 都在 RT-DETRv4-main 下面，相对路径直接这么写
    test_data_root = './data/Processed_Leaf_Dataset'

    print(f"正在检查数据路径: {os.path.abspath(test_data_root)}")

    if os.path.exists(test_data_root):
        # 实例化验证
        train_transforms = get_transforms(split='train', img_size=256)
        dataset = LeafMultiTaskDataset(data_root=test_data_root, split='train', transform=train_transforms)

        if len(dataset) > 0:
            # 抓取第一条数据测试
            img, labels = dataset[0]
            print("\n✅ --- 数据管道测试完美通过！---")
            print(f"输出的图片 Tensor 形状: {img.shape}")
            print(f"输出的标签字典: {labels}")
        else:
            print("⚠️ 数据集为空，请检查 json 文件里是否有内容。")
    else:
        print("❌ 找不到路径！请确认你是在 RT-DETRv4-main 目录下运行的这个脚本。")