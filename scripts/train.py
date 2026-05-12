import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import time
import os
import csv
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from dataset_leaf import LeafMultiTaskDataset, get_transforms
from model_leaf import MultiTaskLeafClassifier


def extract_valid_preds(outputs, labels, ignore_index=-1):
    """提取有效样本（过滤 -1），返回真实标签、预测类别、类别1的概率（用于PR曲线）"""
    valid_mask = (labels != ignore_index)
    valid_labels = labels[valid_mask].cpu().numpy()
    valid_preds = outputs.argmax(dim=1)[valid_mask].cpu().numpy()
    valid_probs = F.softmax(outputs, dim=1)[:, 1][valid_mask].cpu().numpy()
    return valid_labels, valid_preds, valid_probs


def calc_acc(outputs, labels, ignore_index=-1):
    """训练集专用的快速准确率计算"""
    preds = outputs.argmax(dim=1)
    valid_mask = (labels != ignore_index)
    correct = ((preds == labels) & valid_mask).sum().item()
    valid_total = valid_mask.sum().item()
    return correct, valid_total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动引擎，正在使用设备: {device}")

    # 数据集路径
    data_root = r'D:\Pycharm\Project\mmdet_dataset'
    weight_path = None  # 可改为预训练权重路径

    # 训练超参数（可根据显存调整）
    batch_size = 512  # 增大 batch size，充分利用显存
    num_workers = 32  # 增加数据加载线程数
    num_epochs = 300
    learning_rate = 1e-4  # 若 batch size 增大，可适当增大学习率（如 2e-4）
    save_interval = 20
    resume_checkpoint = None

    print("\n📦 正在加载数据集... (已开启纯物理增强)")
    train_dataset = LeafMultiTaskDataset(data_root, split='train', transform=get_transforms('train'))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, prefetch_factor=2)

    val_dataset = LeafMultiTaskDataset(data_root, split='val', transform=get_transforms('val'))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, prefetch_factor=2)

    model = MultiTaskLeafClassifier(pretrained_path=weight_path, hidden_dim=2048)
    model = model.to(device)

    # 损失函数
    criterion_age = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1)
    criterion_yellow = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1)
    criterion_disease = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    scaler = GradScaler()

    start_epoch = 0
    best_mean_acc = 0.0
    best_metrics_report = {}

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"🔄 正在从 {resume_checkpoint} 恢复...")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            best_mean_acc = checkpoint.get('best_mean_acc', 0.0)

    os.makedirs('./weights/checkpoints', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)

    csv_path = './logs/training_log_300_full_metrics.csv'
    if start_epoch == 0:
        with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'LR', 'Train_Loss', 'Train_Acc_Age', 'Train_Acc_Yellow', 'Train_Acc_Disease',
                             'Val_Loss', 'Val_Acc_Age', 'Val_Acc_Yellow', 'Val_Acc_Disease',
                             'Val_P_Age', 'Val_R_Age', 'Val_F1_Age',
                             'Val_P_Yellow', 'Val_R_Yellow', 'Val_F1_Yellow',
                             'Val_P_Dis', 'Val_R_Dis', 'Val_F1_Dis', 'Mean_Val_Acc'])

    print(f"\n🔥 对比实验训练正式开始！(强制完整跑满 {num_epochs} 轮)")

    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()

        # 训练阶段
        model.train()
        total_train_loss = 0.0
        train_correct = {'age': 0, 'yellow': 0, 'disease': 0}
        train_total = {'age': 0, 'yellow': 0, 'disease': 0}

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            age_lbl = labels['age'].to(device)
            yellow_lbl = labels['yellow'].to(device)
            disease_lbl = labels['disease'].to(device)

            optimizer.zero_grad()

            with autocast():
                out_age, out_yellow, out_disease = model(images)
                loss_age = criterion_age(out_age, age_lbl)
                loss_yellow = criterion_yellow(out_yellow, yellow_lbl)
                loss_disease = criterion_disease(out_disease, disease_lbl)
                loss = loss_age + loss_yellow + loss_disease

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()

            for task_name, out, lbl in zip(['age', 'yellow', 'disease'],
                                           [out_age, out_yellow, out_disease],
                                           [age_lbl, yellow_lbl, disease_lbl]):
                c, t = calc_acc(out, lbl)
                train_correct[task_name] += c
                train_total[task_name] += t

        avg_train_loss = total_train_loss / len(train_loader)
        train_acc_age = train_correct['age'] / max(train_total['age'], 1)
        train_acc_yellow = train_correct['yellow'] / max(train_total['yellow'], 1)
        train_acc_disease = train_correct['disease'] / max(train_total['disease'], 1)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # 验证阶段
        model.eval()
        total_val_loss = 0.0

        val_targets = {'age': [], 'yellow': [], 'disease': []}
        val_preds = {'age': [], 'yellow': [], 'disease': []}
        val_probs = {'age': [], 'yellow': [], 'disease': []}

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                age_lbl = labels['age'].to(device)
                yellow_lbl = labels['yellow'].to(device)
                disease_lbl = labels['disease'].to(device)

                with autocast():
                    out_age, out_yellow, out_disease = model(images)
                    loss_age = criterion_age(out_age, age_lbl)
                    loss_yellow = criterion_yellow(out_yellow, yellow_lbl)
                    loss_disease = criterion_disease(out_disease, disease_lbl)
                    total_val_loss += (loss_age + loss_yellow + loss_disease).item()

                for task_name, out, lbl in zip(['age', 'yellow', 'disease'],
                                               [out_age, out_yellow, out_disease],
                                               [age_lbl, yellow_lbl, disease_lbl]):
                    t_lbl, p_cls, p_prob = extract_valid_preds(out, lbl)
                    val_targets[task_name].extend(t_lbl)
                    val_preds[task_name].extend(p_cls)
                    val_probs[task_name].extend(p_prob)

        avg_val_loss = total_val_loss / len(val_loader)

        # 计算指标
        metrics = {}
        for task in ['age', 'yellow', 'disease']:
            y_true = np.array(val_targets[task])
            y_pred = np.array(val_preds[task])

            acc = (y_true == y_pred).mean() if len(y_true) > 0 else 0
            p = precision_score(y_true, y_pred, zero_division=0)
            r = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            metrics[task] = {'acc': acc, 'p': p, 'r': r, 'f1': f1}

        current_mean_acc = (metrics['age']['acc'] + metrics['yellow']['acc'] + metrics['disease']['acc']) / 3.0
        epoch_time = time.time() - start_time

        # 打印与保存
        print(f"\n==== Epoch [{epoch + 1}/{num_epochs}] 结束 | 耗时: {epoch_time:.1f}s ====")
        print(
            f"📉 [Train] LR: {current_lr:.6f} | Loss: {avg_train_loss:.4f} | Acc -> Age: {train_acc_age:.1%} | Yellow: {train_acc_yellow:.1%} | Dis: {train_acc_disease:.1%}")
        print(f"📊 [Val]   Loss: {avg_val_loss:.4f} | Mean Acc: {current_mean_acc:.2%}")
        print(
            f"   ├─ 🌱 Age    -> Acc: {metrics['age']['acc']:.1%} | P: {metrics['age']['p']:.1%} | R: {metrics['age']['r']:.1%} | F1: {metrics['age']['f1']:.1%}")
        print(
            f"   ├─ 🍂 Yellow -> Acc: {metrics['yellow']['acc']:.1%} | P: {metrics['yellow']['p']:.1%} | R: {metrics['yellow']['r']:.1%} | F1: {metrics['yellow']['f1']:.1%}")
        print(
            f"   └─ 🦠 Disease-> Acc: {metrics['disease']['acc']:.1%} | P: {metrics['disease']['p']:.1%} | R: {metrics['disease']['r']:.1%} | F1: {metrics['disease']['f1']:.1%}")

        with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f"{current_lr:.6f}",
                             f"{avg_train_loss:.4f}", f"{train_acc_age:.4f}", f"{train_acc_yellow:.4f}",
                             f"{train_acc_disease:.4f}",
                             f"{avg_val_loss:.4f}", f"{metrics['age']['acc']:.4f}", f"{metrics['yellow']['acc']:.4f}",
                             f"{metrics['disease']['acc']:.4f}",
                             f"{metrics['age']['p']:.4f}", f"{metrics['age']['r']:.4f}", f"{metrics['age']['f1']:.4f}",
                             f"{metrics['yellow']['p']:.4f}", f"{metrics['yellow']['r']:.4f}",
                             f"{metrics['yellow']['f1']:.4f}",
                             f"{metrics['disease']['p']:.4f}", f"{metrics['disease']['r']:.4f}",
                             f"{metrics['disease']['f1']:.4f}",
                             f"{current_mean_acc:.4f}"])

        checkpoint_dict = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_mean_acc': max(best_mean_acc, current_mean_acc)
        }

        torch.save(checkpoint_dict, './weights/checkpoints/latest_checkpoint.pth')
        if (epoch + 1) % save_interval == 0:
            torch.save(checkpoint_dict, f'./weights/checkpoints/checkpoint_epoch_{epoch + 1}.pth')

        if current_mean_acc > best_mean_acc:
            best_mean_acc = current_mean_acc
            best_metrics_report = metrics
            best_metrics_report['epoch'] = epoch + 1
            best_metrics_report['mean_acc'] = current_mean_acc
            best_metrics_report['val_loss'] = avg_val_loss

            torch.save(checkpoint_dict, './weights/checkpoints/best_leaf_model.pth')
            print(f"🌟 发现新最佳模型！平均准确率提升至 {best_mean_acc:.2%}，已覆盖保存 best_leaf_model.pth")

            pr_data_path = './logs/best_pr_curve_data.csv'
            with open(pr_data_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Task', 'True_Label', 'Prob_Class1'])
                for task in ['age', 'yellow', 'disease']:
                    for t_lbl, p_prob in zip(val_targets[task], val_probs[task]):
                        writer.writerow([task, t_lbl, p_prob])
            print(f"📈 最佳模型的 PR 曲线原始数据已保存至: {pr_data_path}")

    print("\n" + "=" * 50)
    print("🎉 300 轮对比实验圆满完结！")
    print("=" * 50)


if __name__ == '__main__':
    main()