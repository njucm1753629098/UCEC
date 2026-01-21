import os
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import HeteroData
import pickle
import time
from collections import defaultdict
import gc
import warnings

# 禁用警告
warnings.filterwarnings("ignore", category=FutureWarning)

DATA_PATH = "/root/autodl-tmp/data"
MAPPING_PATH = "/root/autodl-tmp/mappings.pkl"
GRAPH_DATA_PATH = "/root/autodl-tmp/graph_data.pt"

def load_and_preprocess_data():
    print("开始加载数据...")
    start_time = time.time()
    
    # 读取CSV文件
    herb_target = pd.read_csv(os.path.join(DATA_PATH, "herb_target.csv"))
    target_pathway = pd.read_csv(os.path.join(DATA_PATH, "target_pathway.csv"))
    pathway_disease = pd.read_csv(os.path.join(DATA_PATH, "pathway_disease.csv"))
    symptom_disease = pd.read_csv(os.path.join(DATA_PATH, "symptom_disease.csv"))
    
    print(f"数据加载完成: 耗时 {time.time()-start_time:.2f}秒")
    print(f"中药-靶点: {len(herb_target)}行, 靶点-通路: {len(target_pathway)}行, "
          f"通路-疾病: {len(pathway_disease)}行, 症状-疾病: {len(symptom_disease)}行")
    
    # 创建映射字典
    print("创建节点映射...")
    mapping_time = time.time()
    
    # 中药映射 (使用Chinese Character)
    herb_names = herb_target['Chinese Character'].unique()
    herb_mapping = {name: idx for idx, name in enumerate(herb_names)}
    herb_id_to_name = {row['Herb ID']: row['Chinese Character'] for _, row in herb_target.iterrows()}
    num_herbs = len(herb_mapping)
    print(f"中药节点: {num_herbs}")
    
    # 靶点映射 (使用Gene Symbol)
    target_symbols = herb_target['Gene Symbol'].dropna().unique()
    target_mapping = {symbol: idx for idx, symbol in enumerate(target_symbols)}
    num_targets = len(target_mapping)
    print(f"靶点节点: {num_targets}")
    
    # 通路映射
    pathway_ids = pd.concat([
        target_pathway['Pathway_ID'].dropna(), 
        pathway_disease['Pathway_ID'].dropna()
    ]).unique()
    pathway_mapping = {pid: idx for idx, pid in enumerate(pathway_ids)}
    num_pathways = len(pathway_mapping)
    print(f"通路节点: {num_pathways}")
    
    # 疾病映射
    disease_ids = pd.concat([
        pathway_disease['Disease_MeSHID'].dropna(), 
        symptom_disease['Disease_MeSHID'].dropna()
    ]).unique()
    disease_mapping = {did: idx for idx, did in enumerate(disease_ids)}
    num_diseases = len(disease_mapping)
    print(f"疾病节点: {num_diseases}")
    
    # 症状映射 (使用Symptom_Name)
    symptom_names = symptom_disease['Symptom_Name'].dropna().unique()
    symptom_mapping = {name: idx for idx, name in enumerate(symptom_names)}
    num_symptoms = len(symptom_mapping)
    print(f"症状节点: {num_symptoms}")
    
    print(f"节点映射创建完成: 耗时 {time.time()-mapping_time:.2f}秒")
    
    # 创建异质图
    print("构建异质图...")
    graph_time = time.time()
    data = HeteroData()
    
    # 添加节点
    data['herb'].node_id = torch.arange(num_herbs)
    data['target'].node_id = torch.arange(num_targets)
    data['pathway'].node_id = torch.arange(num_pathways)
    data['disease'].node_id = torch.arange(num_diseases)
    data['symptom'].node_id = torch.arange(num_symptoms)
    
    # 添加边索引 - 更宽松的过滤条件
    print("添加边关系...")
    
    # 1. 中药-靶点边
    herb_target_edges = []
    herb_target_set = set()  # 用于去重
    
    print(f"处理中药-靶点关系 ({len(herb_target)}行)...")
    valid_count = 0
    invalid_count = 0
    
    for _, row in herb_target.iterrows():
        herb_name = row['Chinese Character']
        target_symbol = row['Gene Symbol']
        
        # 跳过缺失值
        if pd.isna(herb_name) or pd.isna(target_symbol):
            invalid_count += 1
            continue
            
        # 获取索引 - 创建新映射如果不存在
        if herb_name not in herb_mapping:
            # 添加新中药节点
            new_idx = len(herb_mapping)
            herb_mapping[herb_name] = new_idx
            num_herbs += 1
            
        if target_symbol not in target_mapping:
            # 添加新靶点节点
            new_idx = len(target_mapping)
            target_mapping[target_symbol] = new_idx
            num_targets += 1
            
        herb_idx = herb_mapping[herb_name]
        target_idx = target_mapping[target_symbol]
        
        # 去重
        edge_key = (herb_idx, target_idx)
        if edge_key not in herb_target_set:
            herb_target_edges.append([herb_idx, target_idx])
            herb_target_set.add(edge_key)
            valid_count += 1
    
    print(f"  有效边: {valid_count}, 无效边: {invalid_count}")
    
    # 更新节点ID张量
    data['herb'].node_id = torch.arange(len(herb_mapping))
    data['target'].node_id = torch.arange(len(target_mapping))
    
    herb_target_edges = torch.tensor(herb_target_edges).t().contiguous()
    data['herb', 'affects', 'target'].edge_index = herb_target_edges
    
    # 2. 靶点-通路边
    target_pathway_edges = []
    target_pathway_set = set()
    
    # 创建靶点ID到Gene Symbol的映射
    target_id_to_symbol = {}
    for _, row in herb_target.iterrows():
        if not pd.isna(row['Uniprot ID']) and not pd.isna(row['Gene Symbol']):
            target_id_to_symbol[row['Uniprot ID']] = row['Gene Symbol']
    
    print(f"处理靶点-通路关系 ({len(target_pathway)}行)...")
    valid_count = 0
    invalid_count = 0
    
    for _, row in target_pathway.iterrows():
        target_id = row['Target_UniProtID']
        pathway_id = row['Pathway_ID']
        
        # 跳过缺失值
        if pd.isna(target_id) or pd.isna(pathway_id):
            invalid_count += 1
            continue
            
        # 获取靶点符号
        if target_id not in target_id_to_symbol:
            invalid_count += 1
            continue
            
        target_symbol = target_id_to_symbol[target_id]
        
        # 获取索引 - 创建新映射如果不存在
        if target_symbol not in target_mapping:
            # 添加新靶点节点
            new_idx = len(target_mapping)
            target_mapping[target_symbol] = new_idx
            num_targets += 1
            
        if pathway_id not in pathway_mapping:
            # 添加新通路节点
            new_idx = len(pathway_mapping)
            pathway_mapping[pathway_id] = new_idx
            num_pathways += 1
            
        target_idx = target_mapping[target_symbol]
        pathway_idx = pathway_mapping[pathway_id]
        
        # 去重
        edge_key = (target_idx, pathway_idx)
        if edge_key not in target_pathway_set:
            target_pathway_edges.append([target_idx, pathway_idx])
            target_pathway_set.add(edge_key)
            valid_count += 1
    
    print(f"  有效边: {valid_count}, 无效边: {invalid_count}")
    
    # 更新节点ID张量
    data['target'].node_id = torch.arange(len(target_mapping))
    data['pathway'].node_id = torch.arange(len(pathway_mapping))
    
    target_pathway_edges = torch.tensor(target_pathway_edges).t().contiguous()
    data['target', 'involved_in', 'pathway'].edge_index = target_pathway_edges
    
    # 3. 通路-疾病边
    pathway_disease_edges = []
    pathway_disease_set = set()
    
    print(f"处理通路-疾病关系 ({len(pathway_disease)}行)...")
    valid_count = 0
    invalid_count = 0
    
    for _, row in pathway_disease.iterrows():
        pathway_id = row['Pathway_ID']
        disease_id = row['Disease_MeSHID']
        
        # 跳过缺失值
        if pd.isna(pathway_id) or pd.isna(disease_id):
            invalid_count += 1
            continue
            
        # 获取索引 - 创建新映射如果不存在
        if pathway_id not in pathway_mapping:
            # 添加新通路节点
            new_idx = len(pathway_mapping)
            pathway_mapping[pathway_id] = new_idx
            num_pathways += 1
            
        if disease_id not in disease_mapping:
            # 添加新疾病节点
            new_idx = len(disease_mapping)
            disease_mapping[disease_id] = new_idx
            num_diseases += 1
            
        pathway_idx = pathway_mapping[pathway_id]
        disease_idx = disease_mapping[disease_id]
        
        # 去重
        edge_key = (pathway_idx, disease_idx)
        if edge_key not in pathway_disease_set:
            pathway_disease_edges.append([pathway_idx, disease_idx])
            pathway_disease_set.add(edge_key)
            valid_count += 1
    
    print(f"  有效边: {valid_count}, 无效边: {invalid_count}")
    
    # 更新节点ID张量
    data['pathway'].node_id = torch.arange(len(pathway_mapping))
    data['disease'].node_id = torch.arange(len(disease_mapping))
    
    pathway_disease_edges = torch.tensor(pathway_disease_edges).t().contiguous()
    data['pathway', 'associated_with', 'disease'].edge_index = pathway_disease_edges
    
    # 4. 症状-疾病边
    symptom_disease_edges = []
    symptom_disease_set = set()
    
    print(f"处理症状-疾病关系 ({len(symptom_disease)}行)...")
    valid_count = 0
    invalid_count = 0
    
    for _, row in symptom_disease.iterrows():
        symptom_name = row['Symptom_Name']
        disease_id = row['Disease_MeSHID']
        
        # 跳过缺失值
        if pd.isna(symptom_name) or pd.isna(disease_id):
            invalid_count += 1
            continue
            
        # 获取索引 - 创建新映射如果不存在
        if symptom_name not in symptom_mapping:
            # 添加新症状节点
            new_idx = len(symptom_mapping)
            symptom_mapping[symptom_name] = new_idx
            num_symptoms += 1
            
        if disease_id not in disease_mapping:
            # 添加新疾病节点
            new_idx = len(disease_mapping)
            disease_mapping[disease_id] = new_idx
            num_diseases += 1
            
        symptom_idx = symptom_mapping[symptom_name]
        disease_idx = disease_mapping[disease_id]
        
        # 去重
        edge_key = (symptom_idx, disease_idx)
        if edge_key not in symptom_disease_set:
            symptom_disease_edges.append([symptom_idx, disease_idx])
            symptom_disease_set.add(edge_key)
            valid_count += 1
    
    print(f"  有效边: {valid_count}, 无效边: {invalid_count}")
    
    # 更新节点ID张量
    data['symptom'].node_id = torch.arange(len(symptom_mapping))
    data['disease'].node_id = torch.arange(len(disease_mapping))
    
    symptom_disease_edges = torch.tensor(symptom_disease_edges).t().contiguous()
    data['symptom', 'indicates', 'disease'].edge_index = symptom_disease_edges
    
    # 手动添加反向边
    print("添加反向边...")
    
    # 中药-靶点反向边
    rev_herb_target = data['herb', 'affects', 'target'].edge_index[[1, 0]].contiguous()
    data['target', 'rev_affects', 'herb'].edge_index = rev_herb_target
    
    # 靶点-通路反向边
    rev_target_pathway = data['target', 'involved_in', 'pathway'].edge_index[[1, 0]].contiguous()
    data['pathway', 'rev_involved_in', 'target'].edge_index = rev_target_pathway
    
    # 通路-疾病反向边
    rev_pathway_disease = data['pathway', 'associated_with', 'disease'].edge_index[[1, 0]].contiguous()
    data['disease', 'rev_associated_with', 'pathway'].edge_index = rev_pathway_disease
    
    # 症状-疾病反向边
    rev_symptom_disease = data['symptom', 'indicates', 'disease'].edge_index[[1, 0]].contiguous()
    data['disease', 'rev_indicates', 'symptom'].edge_index = rev_symptom_disease
    
    # 保存映射关系
    mappings = {
        'herb': herb_mapping,
        'herb_id_to_name': herb_id_to_name,
        'target': target_mapping,
        'pathway': pathway_mapping,
        'disease': disease_mapping,
        'symptom': symptom_mapping,
        'reverse_herb': {v: k for k, v in herb_mapping.items()},
        'reverse_target': {v: k for k, v in target_mapping.items()},
        'reverse_pathway': {v: k for k, v in pathway_mapping.items()},
        'reverse_disease': {v: k for k, v in disease_mapping.items()},
        'reverse_symptom': {v: k for k, v in symptom_mapping.items()},
    }
    
    with open(MAPPING_PATH, 'wb') as f:
        pickle.dump(mappings, f)
    
    # 保存图数据
    torch.save(data, GRAPH_DATA_PATH)
    
    print(f"图构建完成: 总耗时 {time.time()-graph_time:.2f}秒")
    print(f"中药节点: {len(herb_mapping)}, 靶点节点: {len(target_mapping)}, "
          f"通路节点: {len(pathway_mapping)}, 疾病节点: {len(disease_mapping)}, "
          f"症状节点: {len(symptom_mapping)}")
    print(f"边数量: 中药-靶点: {herb_target_edges.size(1)}, 靶点-通路: {target_pathway_edges.size(1)}, "
          f"通路-疾病: {pathway_disease_edges.size(1)}, 症状-疾病: {symptom_disease_edges.size(1)}")
    
    # 清理内存
    del herb_target, target_pathway, pathway_disease, symptom_disease
    gc.collect()
    
    return data, mappings

def load_data():
    # 尝试加载已有的映射和图数据
    if os.path.exists(GRAPH_DATA_PATH) and os.path.exists(MAPPING_PATH):
        print("加载已保存的图数据和映射...")
        data = torch.load(GRAPH_DATA_PATH)
        with open(MAPPING_PATH, 'rb') as f:
            mappings = pickle.load(f)
        return data, mappings
    else:
        return load_and_preprocess_data()