import torch
import torch.optim as optim
import torch.nn.functional as F
from data_loader import load_data
from model import Stage1Model, Stage2Model
from utils import save_stage2_progress, find_paths, is_original_path, generate_positive_samples
import time
import os
import random
import numpy as np
from tqdm import tqdm
import copy
import gc

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

def corrupt_path(data, path, mappings):
    """创建负样本路径：随机替换路径中的一个节点"""
    corrupted_path = copy.deepcopy(path)
    node_types = ['target', 'pathway', 'disease', 'symptom']
    corrupt_type = random.choice(node_types)
    
    if corrupt_type == 'target':
        all_targets = list(mappings['reverse_target'].keys())
        all_targets.remove(path['target'])
        if all_targets:
            corrupted_path['target'] = random.choice(all_targets)
    elif corrupt_type == 'pathway':
        all_pathways = list(mappings['reverse_pathway'].keys())
        all_pathways.remove(path['pathway'])
        if all_pathways:
            corrupted_path['pathway'] = random.choice(all_pathways)
    elif corrupt_type == 'disease':
        all_diseases = list(mappings['reverse_disease'].keys())
        all_diseases.remove(path['disease'])
        if all_diseases:
            corrupted_path['disease'] = random.choice(all_diseases)
    elif corrupt_type == 'symptom':
        all_symptoms = list(mappings['reverse_symptom'].keys())
        all_symptoms.remove(path['symptom'])
        if all_symptoms:
            corrupted_path['symptom'] = random.choice(all_symptoms)
    
    return corrupted_path, corrupt_type

def train_stage2():
    print("="*50)
    print("开始训练阶段2：解释链预测 - Transformer版本")
    print("="*50)
    start_time = time.time()
    
    # 加载数据和阶段1模型
    data, mappings = load_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建阶段1模型时传入data对象
    stage1_model = Stage1Model(data, hidden_channels=128, out_channels=64, num_relations=RELATION_TYPES).to(device)
    stage1_model.load_state_dict(torch.load("best_stage1_model.pth"))
    stage1_model.eval()
    
    # 初始化阶段2模型
    model = Stage2Model(hidden_channels=64).to(device)
    data = data.to(device)
    
    # 获取节点嵌入
    with torch.no_grad():
        x_dict = stage1_model(None, data.edge_index_dict)
    
    # 保存嵌入供后续使用
    model.x_dict = x_dict
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 训练循环
    num_epochs = 100  # 增加训练轮次
    batch_size = 64    # 批量大小
    print(f"开始训练，共 {num_epochs} 个epoch...")
    progress_file = "stage2_training_progress.csv"
    
    # 创建日志文件并写入标题
    with open(progress_file, 'w') as f:
        f.write("epoch,loss,pos_loss,neg_loss,acc,learning_rate,epoch_time\n")
    
    best_loss = float('inf')
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        
        total_loss = 0
        total_pos_loss = 0
        total_neg_loss = 0
        total_acc = 0
        total_count = 0
        
        # 随机选择中药
        herb_indices = random.sample(
            range(len(mappings['herb'])), 
            min(batch_size * 10, len(mappings['herb']))
        )
        
        # 使用tqdm显示进度条
        pbar = tqdm(range(0, len(herb_indices), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for start_idx in pbar:
            optimizer.zero_grad()
            
            batch_loss = 0
            batch_pos_loss = 0
            batch_neg_loss = 0
            batch_acc = 0
            batch_count = 0
            
            # 处理每个批次
            for i in range(start_idx, min(start_idx + batch_size, len(herb_indices))):
                herb_idx = herb_indices[i]
                
                # 获取该中药的top症状
                with torch.no_grad():
                    scores = stage1_model.predict_herb_symptom(x_dict)
                    top_symptoms = scores[herb_idx].argsort(descending=True)[:5]  # 取前5个症状
                
                for symptom_idx in top_symptoms:
                    symptom_idx = symptom_idx.item()
                    # 获取路径 (允许生成新路径)
                    paths = find_paths(data, herb_idx, symptom_idx, model, mappings, k=3, generate_new=True)
                    
                    if not paths:
                        continue
                    
                    # 只使用最佳路径作为正样本
                    true_path, true_score = paths[0]
                    
                    # 创建负样本路径 - 两种类型
                    neg_path_corrupt, _ = corrupt_path(data, true_path, mappings)
                    
                    # 类型2: 生成的新路径 (确保不在原始图中)
                    neg_path_new = None
                    new_paths = find_paths(data, herb_idx, symptom_idx, model, mappings, k=1, generate_new=True)
                    if new_paths and not is_original_path(new_paths[0][0], data):
                        neg_path_new = new_paths[0][0]
                    
                    # 准备正样本嵌入和节点类型
                    emb_herb = model.x_dict['herb'][true_path['herb']]
                    emb_target = model.x_dict['target'][true_path['target']]
                    emb_pathway = model.x_dict['pathway'][true_path['pathway']]
                    emb_disease = model.x_dict['disease'][true_path['disease']]
                    emb_symptom = model.x_dict['symptom'][true_path['symptom']]
                    true_path_emb = torch.stack([
                        emb_herb, emb_target, emb_pathway, emb_disease, emb_symptom
                    ]).unsqueeze(0)  # [1, 5, hidden]
                    
                    # 节点类型：0:herb, 1:target, 2:pathway, 3:disease, 4:symptom
                    true_node_types = torch.tensor([[0, 1, 2, 3, 4]]).to(device)
                    
                    # 计算正样本损失
                    true_pred_score = model.score_path(true_path_emb, true_node_types)
                    pos_loss = F.binary_cross_entropy(true_pred_score, torch.ones_like(true_pred_score))
                    
                    # 负样本损失 (破坏的路径)
                    emb_herb_neg1 = model.x_dict['herb'][neg_path_corrupt['herb']]
                    emb_target_neg1 = model.x_dict['target'][neg_path_corrupt['target']]
                    emb_pathway_neg1 = model.x_dict['pathway'][neg_path_corrupt['pathway']]
                    emb_disease_neg1 = model.x_dict['disease'][neg_path_corrupt['disease']]
                    emb_symptom_neg1 = model.x_dict['symptom'][neg_path_corrupt['symptom']]
                    neg_path_emb1 = torch.stack([
                        emb_herb_neg1, emb_target_neg1, emb_pathway_neg1, emb_disease_neg1, emb_symptom_neg1
                    ]).unsqueeze(0)
                    
                    neg_node_types1 = torch.tensor([[0, 1, 2, 3, 4]]).to(device)
                    neg_pred_score1 = model.score_path(neg_path_emb1, neg_node_types1)
                    neg_loss1 = F.binary_cross_entropy(neg_pred_score1, torch.zeros_like(neg_pred_score1))
                    
                    # 负样本损失 (新路径)
                    if neg_path_new:
                        emb_herb_neg2 = model.x_dict['herb'][neg_path_new['herb']]
                        emb_target_neg2 = model.x_dict['target'][neg_path_new['target']]
                        emb_pathway_neg2 = model.x_dict['pathway'][neg_path_new['pathway']]
                        emb_disease_neg2 = model.x_dict['disease'][neg_path_new['disease']]
                        emb_symptom_neg2 = model.x_dict['symptom'][neg_path_new['symptom']]
                        neg_path_emb2 = torch.stack([
                            emb_herb_neg2, emb_target_neg2, emb_pathway_neg2, emb_disease_neg2, emb_symptom_neg2
                        ]).unsqueeze(0)
                        
                        neg_node_types2 = torch.tensor([[0, 1, 2, 3, 4]]).to(device)
                        neg_pred_score2 = model.score_path(neg_path_emb2, neg_node_types2)
                        neg_loss2 = F.binary_cross_entropy(neg_pred_score2, torch.zeros_like(neg_pred_score2))
                    else:
                        neg_loss2 = torch.tensor(0.0, device=device)
                    
                    # 组合损失
                    path_loss = pos_loss + neg_loss1 + neg_loss2
                    batch_loss += path_loss.item()
                    batch_pos_loss += pos_loss.item()
                    batch_neg_loss += (neg_loss1.item() + neg_loss2.item()) / 2
                    
                    # 计算准确率
                    batch_acc += (true_pred_score > 0.5).float().mean().item()
                    batch_acc += (neg_pred_score1 < 0.5).float().mean().item()
                    if neg_path_new:
                        batch_acc += (neg_pred_score2 < 0.5).float().mean().item()
                        batch_count += 3
                    else:
                        batch_count += 2
                    
                    # 反向传播
                    path_loss.backward()
            
            # 梯度裁剪防止爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 更新进度条
            if batch_count > 0:
                batch_loss /= batch_count
                batch_acc /= batch_count
                pbar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'acc': f'{batch_acc:.4f}'
                })
                
                total_loss += batch_loss * batch_count
                total_pos_loss += batch_pos_loss
                total_neg_loss += batch_neg_loss * batch_count
                total_acc += batch_acc * batch_count
                total_count += batch_count
        
        # 计算平均损失和准确率
        if total_count > 0:
            total_loss /= total_count
            total_pos_loss /= total_count
            total_neg_loss /= total_count
            total_acc /= total_acc
            
            # 打印进度
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.6f} (Pos: {total_pos_loss:.6f}, Neg: {total_neg_loss:.6f}), Acc: {total_acc:.4f}, Time: {epoch_time:.2f}s")
            
            # 保存训练进度
            save_stage2_progress(epoch+1, total_loss, progress_file, epoch_time, optimizer.param_groups[0]['lr'])
            
            # 保存最佳模型
            if total_loss < best_loss:
                best_loss = total_loss
                torch.save(model.state_dict(), "best_stage2_model.pth")
                print(f"保存新的最佳模型 (Loss={best_loss:.6f})")
            
            # 更新学习率
            scheduler.step(total_loss)
        
        # 每5个epoch清理一次缓存
        if (epoch + 1) % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    # 保存最终模型
    model_path = "stage2_model.pth"
    torch.save(model.state_dict(), model_path)
    total_time = time.time() - start_time
    print(f"阶段2训练完成，总耗时: {total_time:.2f}秒，模型已保存至 {model_path}")
    print("="*50)

if __name__ == "__main__":
    train_stage2()