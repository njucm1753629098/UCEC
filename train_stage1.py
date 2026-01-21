import torch
import torch.optim as optim
import torch.nn.functional as F 
from data_loader import load_data
from model import Stage1Model
import os
import time
from utils import (
    save_training_progress, 
    generate_positive_samples, 
    generate_negative_samples,
    split_dataset,
    evaluate_model
)
import numpy as np
import random
from tqdm import tqdm
import datetime
import csv

# 定义关系类型
RELATION_TYPES = [
    ('herb', 'affects', 'target'),
    ('target', 'involved_in', 'pathway'),
    ('pathway', 'associated_with', 'disease'),
    ('symptom', 'indicates', 'disease'),
    ('target', 'rev_affects', 'herb'),
    ('pathway', 'rev_involved_in', 'target'),
    ('disease', 'rev_associated_with', 'pathway'),
    ('disease', 'rev_indicates', 'symptom')
]

def train_stage1():
    print("="*50)
    print("开始训练阶段1：中药-症状初步关联预测")
    print("="*50)
    start_time = time.time()
    
    # 设置全局随机种子 (保证可重复性)
    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    print(f"全局随机种子: {random_seed}")
    
    # 加载数据
    data, mappings = load_data()
    load_time = time.time() - start_time
    print(f"数据加载耗时: {load_time:.2f}秒")
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = Stage1Model(data, hidden_channels=128, out_channels=64, num_relations=RELATION_TYPES).to(device)
    data = data.to(device)
    
    # 生成正样本标签
    pos_start = time.time()
    positive_labels = generate_positive_samples(data, mappings)
    pos_time = time.time() - pos_start
    print(f"正样本生成耗时: {pos_time:.2f}秒")
    
    # 负样本参数设置 - 增加负样本比例和数据增强
    neg_ratio = 1.0  # 增加到5倍负样本
    neg_augment_factor = 20  # 20倍增强
    
    # 生成负样本 - 启用20倍增强
    neg_start = time.time()
    num_herbs = len(mappings['herb'])
    num_symptoms = len(mappings['symptom'])
    neg_edge_index = generate_negative_samples(
        positive_labels, 
        num_herbs, 
        num_symptoms, 
        neg_ratio=neg_ratio,
        augment_factor=neg_augment_factor
    )
    neg_time = time.time() - neg_start
    print(f"负样本生成耗时: {neg_time:.2f}秒")
    
    # 计算样本比例
    pos_count = positive_labels.sum().item()
    neg_count = neg_edge_index.size(1)
    total_samples = pos_count + neg_count
    pos_ratio = pos_count / total_samples * 100
    neg_ratio_val = neg_count / total_samples * 100

    print(f"正样本数量: {pos_count}, 负样本数量: {neg_count} (增强 {neg_augment_factor}x)")
    print(f"样本分布: 正样本 {pos_ratio:.2f}%, 负样本 {neg_ratio_val:.2f}%")
    print(f"正负样本比例: 1:{neg_count/pos_count:.2f}")
    
    # 划分训练集和测试集
    test_size = 0.2  # 20%测试集
    train_pos, train_neg, test_pos, test_neg = split_dataset(
        positive_labels, 
        neg_edge_index, 
        test_size=test_size, 
        random_seed=random_seed
    )
    
    # 转换为张量并移动到设备
    positive_labels = positive_labels.to(device)
    train_pos = train_pos.to(device)
    train_neg = train_neg.to(device)
    test_pos = test_pos.to(device)
    test_neg = test_neg.to(device)
    
    # 优化器 - 增加权重衰减
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)  # 增加正则化
    
    # 使用自定义的学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # 训练循环
    num_epochs = 200  # 增加epoch数量
    print(f"开始训练，共 {num_epochs} 个epoch...")
    
    # 创建带时间戳的进度文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    progress_file = f"stage1_training_progress_{timestamp}.csv"
    
    # 创建日志文件并写入标题
    with open(progress_file, 'w') as f:
        f.write("epoch,loss,pos_loss,neg_loss,acc,pos_acc,neg_acc,lr,time,test_acc,test_pos_acc,test_neg_acc,test_auc,test_f1\n")
    
    best_loss = float('inf')
    best_f1 = 0.0
    patience = 200  # 早停机制
    epochs_no_improve = 0
    
    # 添加进度条
    pbar = tqdm(range(num_epochs), desc="训练阶段1")
    
    for epoch in pbar:
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        x_dict = model(None, data.edge_index_dict)
        
        # 预测分数
        scores = model.predict_herb_symptom(x_dict)
        
        # 计算训练集正样本损失
        train_pos_preds = scores[train_pos[0], train_pos[1]]
        pos_loss = F.binary_cross_entropy(
            train_pos_preds, 
            torch.ones_like(train_pos_preds),
            reduction='mean'
        )
        
        # 计算训练集负样本损失
        train_neg_preds = scores[train_neg[0], train_neg[1]]
        neg_loss = F.binary_cross_entropy(
            train_neg_preds, 
            torch.zeros_like(train_neg_preds),
            reduction='mean'
        )
        
        # 组合损失 - 平衡正负样本
        loss = pos_loss + neg_loss
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪防止爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # 计算训练集准确率
        with torch.no_grad():
            train_pos_acc = (train_pos_preds > 0.5).float().mean().item()
            train_neg_acc = (train_neg_preds < 0.5).float().mean().item()
            train_total_acc = (train_pos_acc * len(train_pos_preds) + train_neg_acc * len(train_neg_preds)) / (len(train_pos_preds) + len(train_neg_preds))
        
        # 在测试集上评估模型
        test_metrics = evaluate_model(model, test_pos, test_neg)
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 计算epoch时间
        epoch_time = time.time() - epoch_start
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'train_acc': f'{train_total_acc:.4f}',
            'test_acc': f'{test_metrics["acc"]:.4f}',
            'test_f1': f'{test_metrics["f1"]:.4f}'
        })
        
        # 保存训练进度 (包含训练和测试指标)
        save_training_progress(
            epoch+1, 
            loss.item(), 
            pos_loss.item(), 
            neg_loss.item(), 
            train_total_acc,
            train_pos_acc,
            train_neg_acc,
            current_lr,
            epoch_time,
            progress_file,
            test_metrics
        )
        
        # 每10个epoch打印详细指标
        if (epoch + 1) % 10 == 0 or (epoch + 1) == 1 or (epoch + 1) == num_epochs:
            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            print(f"  [训练集] Loss: {loss.item():.6f} (Pos: {pos_loss.item():.6f}, Neg: {neg_loss.item():.6f})")
            print(f"  [训练集] Accuracy: Total {train_total_acc:.4f}, Pos {train_pos_acc:.4f}, Neg {train_neg_acc:.4f}")
            print(f"  [测试集] Accuracy: Total {test_metrics['acc']:.4f}, Pos {test_metrics['pos_acc']:.4f}, Neg {test_metrics['neg_acc']:.4f}")
            print(f"  [测试集] AUC: {test_metrics['auc']:.4f}, F1: {test_metrics['f1']:.4f}, Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}, Time: {epoch_time:.2f}s")
        
        # 保存最佳模型 (基于测试集F1分数)
        if test_metrics['f1'] > best_f1:
            best_f1 = test_metrics['f1']
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_stage1_model.pth")
            print(f"保存新的最佳模型 (F1={best_f1:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n早停: 连续 {patience} 个epoch没有改进")
                break
    
    # 保存最终模型
    model_path = "stage1_model.pth"
    torch.save(model.state_dict(), model_path)
    total_time = time.time() - start_time
    
    # 最终评估
    final_test_metrics = evaluate_model(model, test_pos, test_neg)
    print("\n" + "="*50)
    print("最终测试集性能:")
    print(f"  准确率: {final_test_metrics['acc']:.4f}")
    print(f"  正样本准确率: {final_test_metrics['pos_acc']:.4f}")
    print(f"  负样本准确率: {final_test_metrics['neg_acc']:.4f}")
    print(f"  AUC: {final_test_metrics['auc']:.4f}")
    print(f"  F1分数: {final_test_metrics['f1']:.4f}")
    print(f"  精确率: {final_test_metrics['precision']:.4f}")
    print(f"  召回率: {final_test_metrics['recall']:.4f}")
    print("="*50)
    
    print(f"\n阶段1训练完成，总耗时: {total_time/60:.2f}分钟，模型已保存至 {model_path}")
    print(f"训练日志已保存至: {progress_file}")
    print(f"最佳模型 (F1={best_f1:.4f}) 已保存为 best_stage1_model.pth")
    print("="*50)


def save_training_progress(epoch, loss, pos_loss, neg_loss, acc, pos_acc, neg_acc, lr, epoch_time, filename, test_metrics=None):
    """保存训练进度（包含测试指标）"""
    with open(filename, 'a') as f:
        writer = csv.writer(f)
        if test_metrics is None:
            # 只写入训练指标
            writer.writerow([
                epoch, loss, pos_loss, neg_loss, acc, 
                pos_acc, neg_acc, lr, epoch_time,
                "", "", "", "", ""  # 测试指标占位
            ])
        else:
            # 写入训练和测试指标
            writer.writerow([
                epoch, loss, pos_loss, neg_loss, acc, 
                pos_acc, neg_acc, lr, epoch_time,
                test_metrics['acc'], test_metrics['pos_acc'], test_metrics['neg_acc'],
                test_metrics['auc'], test_metrics['f1']
            ])

if __name__ == "__main__":
    train_stage1()