import torch
import torch.nn.functional as F  # 修复 F 未定义的问题
import numpy as np
import random
from collections import defaultdict
import torch_geometric.utils as tg_utils
from tqdm import tqdm
import os
import pickle
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import heapq
import time

# 设置绘图风格
sns.set(style="whitegrid", palette="muted", font_scale=1.2)
plt.rcParams["figure.figsize"] = (12, 9)
plt.rcParams["figure.dpi"] = 100

class TorchKNNManager:
    """基于PyTorch的KNN管理器，替代FAISS"""
    def __init__(self, embeddings, device='cuda'):
        self.embeddings = embeddings
        self.device = device
        # 安全处理空嵌入
        if embeddings.size(0) > 0:
            self.norm_embeddings = F.normalize(embeddings, p=2, dim=1)
        else:
            self.norm_embeddings = embeddings
        
    def search(self, query, k=5):
        """使用余弦相似度搜索最近邻"""
        if query.dim() == 1:
            query = query.unsqueeze(0)
        
        query = query.to(self.device)
        norm_query = F.normalize(query, p=2, dim=1)
        
        # 安全处理空嵌入
        if self.norm_embeddings.size(0) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        
        # 计算余弦相似度
        sim_matrix = torch.mm(norm_query, self.norm_embeddings.t())
        
        # 获取topk
        k = min(k, sim_matrix.size(1))
        topk_sim, topk_indices = torch.topk(sim_matrix, k=k, dim=1)
        return topk_indices.cpu().numpy(), topk_sim.cpu().numpy()

def advanced_fusion(s1, path_scores, paths, data):
    """
    创新融合算法：多维度路径融合
    参数：
        s1: 阶段1预测分数
        path_scores: 所有路径分数列表
        paths: 所有路径信息列表
        data: 图数据
    
    返回：
        融合后的最终分数
    """
    if not path_scores:
        return s1 * 0.7  # 无路径时降低置信度
    
    # 1. 路径质量指标
    avg_score = np.mean(path_scores)
    max_score = max(path_scores)
    original_count = sum(1 for path in paths if is_original_path(path, data))
    original_ratio = original_count / len(paths) if paths else 0
    
    # 2. 路径多样性指标
    node_types = ['target', 'pathway', 'disease']
    diversity = sum(len(set(path[t] for path in paths)) for t in node_types) / (len(paths) * len(node_types)) if paths else 0
    
    # 3. 可靠性加权
    reliability = 0.5 * original_ratio + 0.3 * avg_score + 0.2 * diversity
    
    # 4. 分数融合公式
    fused_score = reliability * max_score + (1 - reliability) * s1
    
    # 5. 路径数量增强
    path_count_factor = min(1.0, len(paths) / 5)  # 最大增强20%
    fused_score = min(1.0, fused_score * (1 + 0.2 * path_count_factor))
    
    return fused_score

def dynamic_fusion(s1, s2, alpha=0.7):
    """动态加权融合算法"""
    if isinstance(s2, list):
        s2 = torch.tensor(s2).mean().to(s1.device)
    
    adjusted_alpha = alpha * torch.sigmoid(5 * (s1 - 0.5)) + (1 - alpha) * torch.sigmoid(5 * (s2 - 0.5))
    return adjusted_alpha * s1 + (1 - adjusted_alpha) * s2

def find_paths(data, herb_idx, symptom_idx, model, mappings, k=5, generate_new=True):
    """高效查找前k条解释链 - PyTorch替代FAISS版本"""
    # 使用缓存避免重复计算
    cache_dir = "./path_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"cache_{herb_idx}_{symptom_idx}.pkl")
    
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # 初始化PyTorch KNN索引
    if not hasattr(model, 'torch_indices'):
        model.torch_indices = {}
        for node_type in ['target', 'pathway', 'disease', 'symptom']:
            embeddings = model.x_dict[node_type]
            model.torch_indices[node_type] = TorchKNNManager(embeddings, device=embeddings.device)
    
    # 获取设备信息
    device = data['herb'].node_id.device
    
    # 获取疾病相关的路径（优化查询）
    symptom_disease_edges = data['symptom', 'indicates', 'disease'].edge_index
    disease_indices = set(symptom_disease_edges[1][symptom_disease_edges[0] == symptom_idx].tolist())
    
    # 通路-疾病连接
    pathway_disease_edges = data['pathway', 'associated_with', 'disease'].edge_index
    if disease_indices:
        disease_tensor = torch.tensor(list(disease_indices), device=device)
        pathway_mask = torch.isin(pathway_disease_edges[1], disease_tensor)
        pathway_indices = set(pathway_disease_edges[0][pathway_mask].tolist())
    else:
        pathway_indices = set()
    
    # 靶点-通路连接
    target_pathway_edges = data['target', 'involved_in', 'pathway'].edge_index
    if pathway_indices:
        pathway_tensor = torch.tensor(list(pathway_indices), device=device)
        target_mask = torch.isin(target_pathway_edges[1], pathway_tensor)
        target_indices = set(target_pathway_edges[0][target_mask].tolist())
    else:
        target_indices = set()
    
    # 中药-靶点连接
    herb_target_edges = data['herb', 'affects', 'target'].edge_index
    herb_mask = herb_target_edges[0] == herb_idx
    herb_targets = set(herb_target_edges[1][herb_mask].tolist())
    valid_targets = herb_targets & target_indices
    
    # 构建有效路径
    valid_paths = []
    for target_idx in valid_targets:
        # 获取靶点相关通路
        target_mask = target_pathway_edges[0] == target_idx
        connected_pathways = set(target_pathway_edges[1][target_mask].tolist()) & pathway_indices
        
        for pathway_idx in connected_pathways:
            # 获取通路相关疾病
            pathway_mask = pathway_disease_edges[0] == pathway_idx
            connected_diseases = set(pathway_disease_edges[1][pathway_mask].tolist()) & disease_indices
            
            for disease_idx in connected_diseases:
                # 检查疾病与症状的连接
                disease_mask = symptom_disease_edges[1] == disease_idx
                if symptom_idx in symptom_disease_edges[0][disease_mask].tolist():
                    path = {
                        'herb': herb_idx,
                        'target': target_idx,
                        'pathway': pathway_idx,
                        'disease': disease_idx,
                        'symptom': symptom_idx
                    }
                    valid_paths.append(path)
    
    # 生成新路径（PyTorch加速版）
    if generate_new and len(valid_paths) < k:
        new_paths = generate_new_paths_torch(
            data, herb_idx, symptom_idx, model, mappings, 
            k - len(valid_paths), model.torch_indices
        )
        valid_paths.extend(new_paths)
    
    # 批量计算路径分数
    path_embs = []
    node_types_list = []  # 存储每条路径的节点类型
    for path in valid_paths:
        emb_herb = model.x_dict['herb'][path['herb']]
        emb_target = model.x_dict['target'][path['target']]
        emb_pathway = model.x_dict['pathway'][path['pathway']]
        emb_disease = model.x_dict['disease'][path['disease']]
        emb_symptom = model.x_dict['symptom'][path['symptom']]
        path_embs.append(torch.stack([
            emb_herb, emb_target, emb_pathway, emb_disease, emb_symptom
        ]))
        # 节点类型：0:herb, 1:target, 2:pathway, 3:disease, 4:symptom
        node_types_list.append(torch.tensor([0, 1, 2, 3, 4]))
    
    if path_embs:
        path_embs = torch.stack(path_embs)  # [num_paths, 5, hidden_dim]
        node_types = torch.stack(node_types_list).to(path_embs.device)  # [num_paths, 5]
        with torch.no_grad():
            path_scores = model.score_path(path_embs, node_types).squeeze(1).tolist()
        paths_with_scores = list(zip(valid_paths, path_scores))
        paths_with_scores.sort(key=lambda x: x[1], reverse=True)
        result = paths_with_scores[:k]
        
        # 保存缓存
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        return result
    return []



def generate_new_paths_torch(data, herb_idx, symptom_idx, model, mappings, num_paths, torch_indices):
    """PyTorch加速的新路径生成"""
    new_paths = []
    device = next(model.parameters()).device
    
    # 1. 中药到靶点：相似度搜索
    herb_emb = model.x_dict['herb'][herb_idx].unsqueeze(0)
    candidate_targets, _ = torch_indices['target'].search(herb_emb, k=min(50, len(mappings['target'])))
    candidate_targets = candidate_targets[0] if candidate_targets.size > 0 else []
    
    # 2. 靶点到通路
    for target_idx in candidate_targets:
        target_emb = model.x_dict['target'][target_idx].unsqueeze(0)
        candidate_pathways, _ = torch_indices['pathway'].search(target_emb, k=min(10, len(mappings['pathway'])))
        candidate_pathways = candidate_pathways[0] if candidate_pathways.size > 0 else []
        
        for pathway_idx in candidate_pathways:
            # 3. 通路到疾病
            pathway_emb = model.x_dict['pathway'][pathway_idx].unsqueeze(0)
            candidate_diseases, _ = torch_indices['disease'].search(pathway_emb, k=min(10, len(mappings['disease'])))
            candidate_diseases = candidate_diseases[0] if candidate_diseases.size > 0 else []
            
            for disease_idx in candidate_diseases:
                # 4. 疾病到症状
                disease_emb = model.x_dict['disease'][disease_idx].unsqueeze(0)
                candidate_symptoms, _ = torch_indices['symptom'].search(disease_emb, k=min(5, len(mappings['symptom'])))
                candidate_symptoms = candidate_symptoms[0] if candidate_symptoms.size > 0 else []
                
                if symptom_idx in candidate_symptoms:
                    path = {
                        'herb': herb_idx,
                        'target': target_idx,
                        'pathway': pathway_idx,
                        'disease': disease_idx,
                        'symptom': symptom_idx
                    }
                    if not is_original_path(path, data):
                        new_paths.append(path)
                        if len(new_paths) >= num_paths:
                            return new_paths
    return new_paths

def is_original_path(path, data):
    """检查路径是否在原图中存在"""
    # 1. 中药-靶点边
    herb_target_edges = data['herb', 'affects', 'target'].edge_index.t().tolist()
    h_t_exists = any(
        edge[0] == path['herb'] and edge[1] == path['target']
        for edge in herb_target_edges
    )
    
    # 2. 靶点-通路边
    target_pathway_edges = data['target', 'involved_in', 'pathway'].edge_index.t().tolist()
    t_p_exists = any(
        edge[0] == path['target'] and edge[1] == path['pathway']
        for edge in target_pathway_edges
    )
    
    # 3. 通路-疾病边
    pathway_disease_edges = data['pathway', 'associated_with', 'disease'].edge_index.t().tolist()
    p_d_exists = any(
        edge[0] == path['pathway'] and edge[1] == path['disease']
        for edge in pathway_disease_edges
    )
    
    # 4. 疾病-症状边（使用反向关系）
    disease_symptom_edges = data['disease', 'rev_indicates', 'symptom'].edge_index.t().tolist()
    d_s_exists = any(
        edge[0] == path['disease'] and edge[1] == path['symptom']
        for edge in disease_symptom_edges
    )
    
    return h_t_exists and t_p_exists and p_d_exists and d_s_exists

def save_training_progress(epoch, loss, pos_loss, neg_loss, acc, pos_acc, neg_acc, lr, epoch_time, filename, test_metrics=None):
    """保存阶段1训练进度（包含测试指标）"""
    with open(filename, 'a') as f:
        writer = csv.writer(f)
        if test_metrics is None:
            writer.writerow([
                epoch, loss, pos_loss, neg_loss, acc, 
                pos_acc, neg_acc, lr, epoch_time,
                "", "", "", "", ""
            ])
        else:
            writer.writerow([
                epoch, loss, pos_loss, neg_loss, acc, 
                pos_acc, neg_acc, lr, epoch_time,
                test_metrics['acc'], test_metrics['pos_acc'], test_metrics['neg_acc'],
                test_metrics['auc'], test_metrics['f1']
            ])

def save_stage2_progress(epoch, loss, filename, epoch_time=None, lr=None):
    """保存阶段2训练进度（简化版）"""
    with open(filename, 'a') as f:
        writer = csv.writer(f)
        if os.path.getsize(filename) == 0:
            writer.writerow(["epoch", "loss", "epoch_time", "learning_rate"])
        writer.writerow([epoch, loss, epoch_time if epoch_time else "", lr if lr else ""])

def generate_positive_samples(data, mappings):
    """高效生成正样本（存在完整路径的中药-症状对）"""
    print("生成正样本...")
    num_herbs = len(mappings['herb'])
    num_symptoms = len(mappings['symptom'])
    labels = torch.zeros((num_herbs, num_symptoms), dtype=torch.float32)
    
    # 使用集合和字典加速查找
    herb_target_dict = defaultdict(set)
    for edge in data['herb', 'affects', 'target'].edge_index.t().tolist():
        herb_target_dict[edge[0]].add(edge[1])
    
    target_pathway_dict = defaultdict(set)
    for edge in data['target', 'involved_in', 'pathway'].edge_index.t().tolist():
        target_pathway_dict[edge[0]].add(edge[1])
    
    pathway_disease_dict = defaultdict(set)
    for edge in data['pathway', 'associated_with', 'disease'].edge_index.t().tolist():
        pathway_disease_dict[edge[0]].add(edge[1])
    
    disease_symptom_dict = defaultdict(set)
    for edge in data['symptom', 'indicates', 'disease'].edge_index.t().tolist():
        disease_symptom_dict[edge[1]].add(edge[0])
    
    # 遍历所有疾病
    for disease_idx in range(len(mappings['disease'])):
        symptom_indices = disease_symptom_dict.get(disease_idx, set())
        if not symptom_indices:
            continue
        
        pathway_indices = set()
        for pathway_idx, diseases in pathway_disease_dict.items():
            if disease_idx in diseases:
                pathway_indices.add(pathway_idx)
        
        if not pathway_indices:
            continue
        
        target_indices = set()
        for target_idx, pathways in target_pathway_dict.items():
            if any(pathway in pathway_indices for pathway in pathways):
                target_indices.add(target_idx)
        
        if not target_indices:
            continue
        
        herb_indices = set()
        for herb_idx, targets in herb_target_dict.items():
            if any(target in target_indices for target in targets):
                herb_indices.add(herb_idx)
        
        if not herb_indices:
            continue
        
        for herb_idx in herb_indices:
            for symptom_idx in symptom_indices:
                labels[herb_idx, symptom_idx] = 1.0
    
    pos_count = int(labels.sum().item())
    total = num_herbs * num_symptoms
    print(f"正样本生成完成，正样本数量: {pos_count}, 比例: {pos_count/total*100:.2f}%")
    return labels

def generate_negative_samples(positive_labels, num_herbs, num_symptoms, neg_ratio=1.0, augment_factor=1):
    """生成高质量的负样本 - 支持数据增强版本"""
    print("生成负样本...")
    total_pairs = num_herbs * num_symptoms
    pos_count = int(positive_labels.sum().item())
    max_neg_samples = total_pairs - pos_count
    
    base_neg_samples = min(int(pos_count * neg_ratio), max_neg_samples)
    target_neg_samples = base_neg_samples * augment_factor
    
    print(f"总样本对: {total_pairs}, 正样本: {pos_count}, 最大负样本: {max_neg_samples}")
    print(f"基础负样本: {base_neg_samples}, 增强因子: {augment_factor}x, 目标负样本: {target_neg_samples}")
    
    all_indices = [(i, j) for i in range(num_herbs) for j in range(num_symptoms)]
    neg_candidates = []
    for herb_idx, symptom_idx in all_indices:
        if positive_labels[herb_idx, symptom_idx] == 0:
            neg_candidates.append((herb_idx, symptom_idx))
    
    if len(neg_candidates) < target_neg_samples:
        print(f"警告：负样本空间不足，使用数据增强 ({len(neg_candidates)} → {target_neg_samples})")
        repeat_count = target_neg_samples // len(neg_candidates) + 1
        augmented_neg = neg_candidates * repeat_count
        selected_neg = random.sample(augmented_neg, target_neg_samples)
    else:
        selected_neg = random.sample(neg_candidates, target_neg_samples)
    
    neg_edge_index = torch.tensor(selected_neg).t().contiguous()
    print(f"生成负样本完成: {neg_edge_index.size(1)} 个负样本 (增强 {augment_factor}x)")
    return neg_edge_index

def split_dataset(positive_labels, neg_edge_index, test_size=0.2, random_seed=42):
    """划分数据集为训练集和测试集"""
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    pos_edge_index = torch.nonzero(positive_labels == 1).t()
    pos_indices = pos_edge_index.t().tolist()
    neg_indices = neg_edge_index.t().tolist()
    
    train_pos, test_pos = train_test_split(
        pos_indices, 
        test_size=test_size, 
        random_state=random_seed
    )
    
    train_neg, test_neg = train_test_split(
        neg_indices, 
        test_size=test_size, 
        random_state=random_seed
    )
    
    train_pos_tensor = torch.tensor(train_pos).t().contiguous()
    test_pos_tensor = torch.tensor(test_pos).t().contiguous()
    train_neg_tensor = torch.tensor(train_neg).t().contiguous()
    test_neg_tensor = torch.tensor(test_neg).t().contiguous()
    
    print(f"数据集划分完成 (随机种子={random_seed}, 测试集比例={test_size*100}%)")
    print(f"  训练集: 正样本 {len(train_pos)}, 负样本 {len(train_neg)}")
    print(f"  测试集: 正样本 {len(test_pos)}, 负样本 {len(test_neg)}")
    
    return train_pos_tensor, train_neg_tensor, test_pos_tensor, test_neg_tensor

def evaluate_model(model, test_pos, test_neg):
    """评估模型性能 - 修复AUC计算版本"""
    model.eval()
    with torch.no_grad():
        x_dict = model(None, model.data.edge_index_dict)
        scores = model.predict_herb_symptom(x_dict)
        
        pos_preds = scores[test_pos[0], test_pos[1]]
        neg_preds = scores[test_neg[0], test_neg[1]]
        
        pos_acc = (pos_preds > 0.5).float().mean().item()
        neg_acc = (neg_preds < 0.5).float().mean().item()
        total_acc = (pos_acc * len(pos_preds) + neg_acc * len(neg_preds)) / (len(pos_preds) + len(neg_preds))
        
        all_preds = torch.cat([pos_preds, neg_preds]).cpu().numpy()
        all_labels = torch.cat([
            torch.ones_like(pos_preds), 
            torch.zeros_like(neg_preds)
        ]).cpu().numpy()
        
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except ValueError:
            auc = 0.5
        
        predictions = (all_preds > 0.5).astype(int)
        true_pos = ((predictions == 1) & (all_labels == 1)).sum()
        false_pos = ((predictions == 1) & (all_labels == 0)).sum()
        false_neg = ((predictions == 0) & (all_labels == 1)).sum()
        
        precision = true_pos / (true_pos + false_pos + 1e-10)
        recall = true_pos / (true_pos + false_neg + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        return {
            'loss': None,
            'acc': total_acc,
            'pos_acc': pos_acc,
            'neg_acc': neg_acc,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

def plot_roc_curve(labels, preds, auc_score, output_dir):
    """绘制ROC曲线"""
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('接收者操作特征曲线(ROC)')
    plt.legend(loc="lower right")
    
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label=f'最佳阈值={gmeans[ix]:.2f}')
    
    output_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"ROC曲线已保存至: {output_path}")

def plot_pr_curve(labels, preds, precision, recall, f1, output_dir):
    """绘制PR曲线"""
    precision_curve, recall_curve, _ = precision_recall_curve(labels, preds)
    average_precision = average_precision_score(labels, preds)
    
    plt.figure()
    plt.plot(recall_curve, precision_curve, color='blue', lw=2, 
             label=f'PR曲线 (AP = {average_precision:.4f})')
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('精确率-召回率曲线(PR)')
    plt.legend(loc="upper right")
    
    plt.scatter(recall, precision, marker='o', color='red', 
                label=f'F1分数={f1:.4f}\n精确率={precision:.4f}\n召回率={recall:.4f}')
    
    output_path = os.path.join(output_dir, "pr_curve.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"PR曲线已保存至: {output_path}")

def find_training_log(model_path):
    """查找与模型相关的训练日志"""
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path)
    
    if not model_dir:
        model_dir = "."
    
    if "best" in model_name:
        if os.path.exists(model_dir):
            log_files = [f for f in os.listdir(model_dir) if f.startswith("stage1_training_progress")]
            if log_files:
                return os.path.join(model_dir, sorted(log_files)[-1])
    
    timestamp = model_name.split("_")[-1].split(".")[0]
    if len(timestamp) == 14:
        log_file = f"stage1_training_progress_{timestamp}.csv"
        log_path = os.path.join(model_dir, log_file)
        if os.path.exists(log_path):
            return log_path
    
    return None

def plot_training_curves(log_file, output_dir):
    """从训练日志绘制训练曲线"""
    import pandas as pd
    df = pd.read_csv(log_file)
    
    plt.figure()
    plt.plot(df['epoch'], df['loss'], label='总损失', color='blue')
    plt.plot(df['epoch'], df['pos_loss'], label='正样本损失', color='green', linestyle='--')
    plt.plot(df['epoch'], df['neg_loss'], label='负样本损失', color='red', linestyle='--')
    plt.xlabel('训练轮次')
    plt.ylabel('损失值')
    plt.title('训练损失曲线')
    plt.legend()
    plt.grid(True)
    output_path = os.path.join(output_dir, "training_loss.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"训练损失曲线已保存至: {output_path}")
    
    plt.figure()
    plt.plot(df['epoch'], df['acc'], label='总准确率', color='blue')
    plt.plot(df['epoch'], df['pos_acc'], label='正样本准确率', color='green', linestyle='--')
    plt.plot(df['epoch'], df['neg_acc'], label='负样本准确率', color='red', linestyle='--')
    plt.plot(df['epoch'], df['test_acc'], label='测试集准确率', color='purple')
    plt.xlabel('训练轮次')
    plt.ylabel('准确率')
    plt.title('训练准确率曲线')
    plt.legend()
    plt.grid(True)
    output_path = os.path.join(output_dir, "training_accuracy.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"训练准确率曲线已保存至: {output_path}")
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(df['epoch'], df['test_auc'], label='测试集AUC', color='blue')
    plt.ylabel('AUC')
    plt.title('测试集AUC变化曲线')
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(df['epoch'], df['test_f1'], label='测试集F1分数', color='green')
    plt.xlabel('训练轮次')
    plt.ylabel('F1分数')
    plt.title('测试集F1分数变化曲线')
    plt.grid(True)
    output_path = os.path.join(output_dir, "advanced_metrics.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"高级指标曲线已保存至: {output_path}")