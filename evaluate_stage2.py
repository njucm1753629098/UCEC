import torch
import numpy as np
import os
import sys
import time
import random
from data_loader import load_data
from model import Stage1Model, Stage2Model
from utils import advanced_fusion, find_paths, is_original_path, generate_positive_samples
import pickle
from tqdm import tqdm
import json
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, auc as sk_auc
from sklearn.metrics import precision_score, recall_score, f1_score
import torch.nn.functional as F

# Set plot style
sns.set(style="whitegrid", palette="muted", font_scale=1.2)
plt.rcParams["figure.figsize"] = (12, 9)
plt.rcParams["figure.dpi"] = 100

# Define relation types
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

def evaluate_path_quality(paths_with_scores, data):
    """Evaluate path quality: score, original ratio, diversity"""
    scores = [score for _, score in paths_with_scores] if paths_with_scores else []
    paths = [path for path, _ in paths_with_scores] if paths_with_scores else []
    
    if not scores:
        return {
            'avg_score': 0,
            'max_score': 0,
            'min_score': 0,
            'original_ratio': 0,
            'diversity': {'target': 0, 'pathway': 0, 'disease': 0}
        }
    
    original_count = sum(1 for path in paths if is_original_path(path, data))
    node_types = ['target', 'pathway', 'disease']
    diversity = {
        t: len(set(path[t] for path in paths)) for t in node_types
    }
    
    return {
        'avg_score': np.mean(scores),
        'max_score': max(scores),
        'min_score': min(scores),
        'original_ratio': original_count / len(paths),
        'diversity': diversity
    }

def plot_comparison_curves(scores_stage1, scores_fusion, labels, output_path):
    """Plot ROC and PR curves comparing Stage1 and Fusion scores"""
    plt.figure(figsize=(12, 10))
    
    # ROC curve
    plt.subplot(2, 1, 1)
    fpr1, tpr1, _ = roc_curve(labels, scores_stage1)
    fpr2, tpr2, _ = roc_curve(labels, scores_fusion)
    roc_auc1 = auc(fpr1, tpr1)
    roc_auc2 = auc(fpr2, tpr2)
    
    plt.plot(fpr1, tpr1, 'b-', label=f'Stage1 (AUC = {roc_auc1:.4f})')
    plt.plot(fpr2, tpr2, 'r-', label=f'Fusion (AUC = {roc_auc2:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc="lower right")
    
    # PR curve
    plt.subplot(2, 1, 2)
    precision1, recall1, _ = precision_recall_curve(labels, scores_stage1)
    precision2, recall2, _ = precision_recall_curve(labels, scores_fusion)
    ap1 = average_precision_score(labels, scores_stage1)
    ap2 = average_precision_score(labels, scores_fusion)
    
    plt.plot(recall1, precision1, 'b-', label=f'Stage1 (AP = {ap1:.4f})')
    plt.plot(recall2, precision2, 'r-', label=f'Fusion (AP = {ap2:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison')
    plt.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Comparison curves saved to: {output_path}")

def evaluate_path_predictions(model, data, mappings, eval_pairs):
    """Evaluate overall path predictions using MRR and Hit@k"""
    path_metrics = {
        'total_paths': 0,
        'correct_paths': 0,
        'mrr': [],
        'hit@1': 0,
        'hit@3': 0,
        'hit@5': 0
    }
    
    # For each herb-symptom pair
    for herb_idx, symptom_idx in tqdm(eval_pairs, desc="Evaluating path predictions"):
        # Get true paths (original paths in the graph)
        true_paths = []
        # Get all diseases connected to the symptom
        symptom_disease_edges = data['symptom', 'indicates', 'disease'].edge_index.t().tolist()
        disease_indices = set()
        for edge in symptom_disease_edges:
            if edge[0] == symptom_idx:
                disease_indices.add(edge[1])
        
        # Get all pathways connected to these diseases
        pathway_disease_edges = data['pathway', 'associated_with', 'disease'].edge_index.t().tolist()
        pathway_indices = set()
        for edge in pathway_disease_edges:
            if edge[1] in disease_indices:
                pathway_indices.add(edge[0])
        
        # Get all targets connected to these pathways
        target_pathway_edges = data['target', 'involved_in', 'pathway'].edge_index.t().tolist()
        target_indices = set()
        for edge in target_pathway_edges:
            if edge[1] in pathway_indices:
                target_indices.add(edge[0])
        
        # Get all herbs connected to these targets
        herb_target_edges = data['herb', 'affects', 'target'].edge_index.t().tolist()
        for edge in herb_target_edges:
            if edge[0] == herb_idx and edge[1] in target_indices:
                target_idx = edge[1]
                # For each target, find connected pathways
                for t_p_edge in target_pathway_edges:
                    if t_p_edge[0] == target_idx and t_p_edge[1] in pathway_indices:
                        pathway_idx = t_p_edge[1]
                        # For each pathway, find connected diseases
                        for p_d_edge in pathway_disease_edges:
                            if p_d_edge[0] == pathway_idx and p_d_edge[1] in disease_indices:
                                disease_idx = p_d_edge[1]
                                # Check disease-symptom connection
                                for s_d_edge in symptom_disease_edges:
                                    if s_d_edge[1] == disease_idx and s_d_edge[0] == symptom_idx:
                                        path = {
                                            'herb': herb_idx,
                                            'target': target_idx,
                                            'pathway': pathway_idx,
                                            'disease': disease_idx,
                                            'symptom': symptom_idx
                                        }
                                        if is_original_path(path, data):
                                            true_paths.append(path)
        
        # If no true paths, skip
        if not true_paths:
            continue
        
        # Generate predicted paths
        predicted_paths = find_paths(data, herb_idx, symptom_idx, model, mappings, k=10, generate_new=True)
        predicted_paths = [path for path, _ in predicted_paths] if predicted_paths else []
        
        # Create candidate paths: true paths + predicted paths
        candidate_paths = true_paths + predicted_paths
        candidate_paths = list({json.dumps(path, sort_keys=True): path for path in candidate_paths}.values())
        
        # Score each candidate path
        candidate_scores = []
        for path in candidate_paths:
            # Get node embeddings
            emb_herb = model.x_dict['herb'][path['herb']]
            emb_target = model.x_dict['target'][path['target']]
            emb_pathway = model.x_dict['pathway'][path['pathway']]
            emb_disease = model.x_dict['disease'][path['disease']]
            emb_symptom = model.x_dict['symptom'][path['symptom']]
            
            # Calculate path score
            with torch.no_grad():
                path_emb = torch.stack([emb_herb, emb_target, emb_pathway, emb_disease, emb_symptom]).unsqueeze(0)
                # Prepare node types: herb=0, target=1, pathway=2, disease=3, symptom=4
                node_types = torch.tensor([0, 1, 2, 3, 4]).unsqueeze(0).to(path_emb.device)
                score = model.score_path(path_emb, node_types).item()
            candidate_scores.append((path, score))
        
        # Sort by score
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate ranks for true paths
        for true_path in true_paths:
            rank = None
            for idx, (candidate_path, _) in enumerate(candidate_scores):
                if candidate_path == true_path:
                    rank = idx + 1
                    break
            
            if rank is not None:
                path_metrics['total_paths'] += 1
                path_metrics['mrr'].append(1.0 / rank)
                
                if rank == 1:
                    path_metrics['hit@1'] += 1
                if rank <= 3:
                    path_metrics['hit@3'] += 1
                if rank <= 5:
                    path_metrics['hit@5'] += 1
    
    # Calculate final metrics
    if path_metrics['total_paths'] > 0:
        path_metrics['mrr'] = np.mean(path_metrics['mrr'])
        path_metrics['hit@1'] /= path_metrics['total_paths']
        path_metrics['hit@3'] /= path_metrics['total_paths']
        path_metrics['hit@5'] /= path_metrics['total_paths']
    
    return path_metrics

def evaluate_stage2():
    """Evaluate Stage2 model performance - comprehensive evaluation"""
    print("=" * 50)
    print("Evaluating Stage2 Model Performance - Path Prediction")
    print("=" * 50)
    start_time = time.time()
    
    # Load data and mappings
    data, mappings = load_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    data = data.to(device)
    
    # Load Stage1 model
    stage1_model = Stage1Model(data, hidden_channels=128, out_channels=64, num_relations=RELATION_TYPES).to(device)
    stage1_model.load_state_dict(torch.load("best_stage1_model.pth"))
    stage1_model.eval()
    
    # Load Stage2 model
    stage2_model = Stage2Model(hidden_channels=64).to(device)
    stage2_model.load_state_dict(torch.load("best_stage2_model.pth"))
    stage2_model.eval()
    
    # Get node embeddings
    with torch.no_grad():
        x_dict = stage1_model(None, data.edge_index_dict)
        scores_stage1 = stage1_model.predict_herb_symptom(x_dict)
    
    # Pass node embeddings to Stage2 model
    stage2_model.x_dict = x_dict
    
    # Generate positive samples
    positive_labels = generate_positive_samples(data, mappings)
    pos_indices = torch.nonzero(positive_labels == 1)
    pos_pairs = [(i[0].item(), i[1].item()) for i in pos_indices]
    
    # Generate negative samples (random pairs) - ensure we have enough negatives
    num_herbs = len(mappings['herb'])
    num_symptoms = len(mappings['symptom'])
    neg_pairs = []
    # We want 200 negative samples, but ensure we don't sample positive ones
    # If we cannot find 200 negatives, we use as many as possible and then repeat
    for _ in range(1000):  # Try up to 1000 times to find negatives
        herb_idx = random.randint(0, num_herbs-1)
        symptom_idx = random.randint(0, num_symptoms-1)
        if positive_labels[herb_idx, symptom_idx] == 0:
            neg_pairs.append((herb_idx, symptom_idx))
            if len(neg_pairs) >= 200:
                break
    
    # If we still don't have enough, use augmentation by repeating
    if len(neg_pairs) < 200:
        # Repeat the existing negatives to reach 200
        neg_pairs = neg_pairs * (200 // len(neg_pairs) + 1)
        neg_pairs = neg_pairs[:200]
    
    # Combine positive and negative samples
    eval_pairs = pos_pairs[:200] + neg_pairs
    print(f"Evaluation samples: {len(eval_pairs)} (Positive: {len(pos_pairs[:200])}, Negative: {len(neg_pairs)})")
    
    # Evaluation metrics
    metrics = {
        'stage1_scores': [],
        'fusion_scores': [],
        'labels': [],
        'path_quality': [],
        'node_prediction': defaultdict(lambda: {'hits': defaultdict(int), 'mrr': []}),
    }
    
    # Node types
    node_types = ['target', 'pathway', 'disease', 'symptom']
    
    # Create output directory
    os.makedirs("stage2_evaluation", exist_ok=True)
    detailed_results = []
    
    # Evaluate each sample pair
    for herb_idx, symptom_idx in tqdm(eval_pairs, desc="Evaluating pairs"):
        herb_name = mappings['reverse_herb'][herb_idx]
        symptom_name = mappings['reverse_symptom'][symptom_idx]
        
        # Get Stage1 score
        stage1_score = scores_stage1[herb_idx, symptom_idx].item()
        metrics['stage1_scores'].append(stage1_score)
        metrics['labels'].append(1 if (herb_idx, symptom_idx) in pos_pairs else 0)
        
        # Find top paths
        paths = find_paths(data, herb_idx, symptom_idx, stage2_model, mappings, k=5, generate_new=True)
        
        # Evaluate path quality
        path_quality = evaluate_path_quality(paths, data)
        metrics['path_quality'].append(path_quality)
        
        # Fusion score
        path_scores = [score for _, score in paths] if paths else [0.0]
        paths_list = [path for path, _ in paths] if paths else []
        fusion_score = advanced_fusion(
            stage1_score,
            path_scores,
            paths_list,
            data
        )
        metrics['fusion_scores'].append(fusion_score)
        
        # Node prediction evaluation (only for positive samples)
        if metrics['labels'][-1] == 1 and paths:
            # Only evaluate the first path
            path, _ = paths[0]
            for node_type in node_types:
                true_node = path[node_type]
                candidate_nodes = list(mappings['reverse_' + node_type].keys())
                
                # Only evaluate if reasonable number of candidates
                if len(candidate_nodes) > 5000:  # Skip large node types
                    continue
                
                candidate_scores = []
                
                # Score each candidate node
                for candidate in random.sample(candidate_nodes, min(100, len(candidate_nodes))):
                    test_path = path.copy()
                    test_path[node_type] = candidate
                    
                    emb_herb = stage2_model.x_dict['herb'][test_path['herb']]
                    emb_target = stage2_model.x_dict['target'][test_path['target']]
                    emb_pathway = stage2_model.x_dict['pathway'][test_path['pathway']]
                    emb_disease = stage2_model.x_dict['disease'][test_path['disease']]
                    emb_symptom = stage2_model.x_dict['symptom'][test_path['symptom']]
                    
                    with torch.no_grad():
                        path_emb = torch.stack([
                            emb_herb, emb_target, emb_pathway, emb_disease, emb_symptom
                        ]).unsqueeze(0)
                        # Prepare node types: herb=0, target=1, pathway=2, disease=3, symptom=4
                        node_types_tensor = torch.tensor([0, 1, 2, 3, 4]).unsqueeze(0).to(path_emb.device)
                        score = stage2_model.score_path(path_emb, node_types_tensor).item()
                    
                    candidate_scores.append((candidate, score))
                
                # Sort by score
                candidate_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Calculate rank
                rank = None
                for idx, (node_id, _) in enumerate(candidate_scores):
                    if node_id == true_node:
                        rank = idx + 1
                        break
                
                if rank is not None:
                    # Update hit rate
                    for k in [1, 3, 10]:
                        if rank <= k:
                            metrics['node_prediction'][node_type]['hits'][f'hit@{k}'] += 1
                    
                    # Update MRR
                    metrics['node_prediction'][node_type]['mrr'].append(1.0 / rank)
    
    # Evaluate overall path predictions
    print("Evaluating overall path predictions...")
    path_pred_metrics = evaluate_path_predictions(stage2_model, data, mappings, eval_pairs[:100])
    
    # Calculate summary metrics
    summary = {}
    
    # Path metrics
    summary['avg_path_score'] = np.mean([pq['avg_score'] for pq in metrics['path_quality']])
    summary['original_path_ratio'] = np.mean([pq['original_ratio'] for pq in metrics['path_quality']])
    summary['target_diversity'] = np.mean([pq['diversity']['target'] for pq in metrics['path_quality']])
    summary['pathway_diversity'] = np.mean([pq['diversity']['pathway'] for pq in metrics['path_quality']])
    summary['disease_diversity'] = np.mean([pq['diversity']['disease'] for pq in metrics['path_quality']])
    
    # Add path prediction metrics
    for key, value in path_pred_metrics.items():
        summary[f'path_{key}'] = value
    
    # Node metrics
    for node_type in node_types:
        total = len(metrics['node_prediction'][node_type]['mrr'])
        if total > 0:
            # Hit rate
            for k in [1, 3, 10]:
                hit_key = f'hit@{k}'
                summary[f'{node_type}_{hit_key}'] = metrics['node_prediction'][node_type]['hits'].get(hit_key, 0) / total
            
            # MRR
            summary[f'{node_type}_mrr'] = np.mean(metrics['node_prediction'][node_type]['mrr'])
    
    # Fusion score evaluation
    if metrics['fusion_scores']:
        # Calculate AUC and AP
        summary['stage1_auc'] = roc_auc_score(metrics['labels'], metrics['stage1_scores'])
        summary['fusion_auc'] = roc_auc_score(metrics['labels'], metrics['fusion_scores'])
        summary['stage1_ap'] = average_precision_score(metrics['labels'], metrics['stage1_scores'])
        summary['fusion_ap'] = average_precision_score(metrics['labels'], metrics['fusion_scores'])
        
        # Calculate additional metrics
        threshold = 0.5
        stage1_preds = [1 if score > threshold else 0 for score in metrics['stage1_scores']]
        fusion_preds = [1 if score > threshold else 0 for score in metrics['fusion_scores']]
        
        summary['stage1_precision'] = precision_score(metrics['labels'], stage1_preds)
        summary['stage1_recall'] = recall_score(metrics['labels'], stage1_preds)
        summary['stage1_f1'] = f1_score(metrics['labels'], stage1_preds)
        
        summary['fusion_precision'] = precision_score(metrics['labels'], fusion_preds)
        summary['fusion_recall'] = recall_score(metrics['labels'], fusion_preds)
        summary['fusion_f1'] = f1_score(metrics['labels'], fusion_preds)
        
        # Plot comparison curves
        plot_comparison_curves(
            metrics['stage1_scores'], 
            metrics['fusion_scores'],
            metrics['labels'],
            "stage2_evaluation/comparison_curves.png"
        )
    
    # Print results
    print("\n" + "=" * 50)
    print("Stage2 Model Evaluation Results:")
    print(f"  Paths evaluated: {path_pred_metrics.get('total_paths', 0)}")
    print(f"  Original path ratio: {summary['original_path_ratio']:.4f}")
    print(f"  Average path score: {summary['avg_path_score']:.4f}")
    print(f"  Path diversity: Target={summary['target_diversity']:.2f}, Pathway={summary['pathway_diversity']:.2f}, Disease={summary['disease_diversity']:.2f}")
    
    print("\nPath Prediction Performance:")
    print(f"  MRR: {summary.get('path_mrr', 0):.4f}")
    print(f"  Hit@1: {summary.get('path_hit@1', 0):.4f}")
    print(f"  Hit@3: {summary.get('path_hit@3', 0):.4f}")
    print(f"  Hit@5: {summary.get('path_hit@5', 0):.4f}")
    
    print("\nHerb-Symptom Prediction Performance:")
    print(f"  Stage1 AUC: {summary.get('stage1_auc', 0):.4f}, AP: {summary.get('stage1_ap', 0):.4f}")
    print(f"  Stage1 Precision: {summary.get('stage1_precision', 0):.4f}, Recall: {summary.get('stage1_recall', 0):.4f}, F1: {summary.get('stage1_f1', 0):.4f}")
    print(f"  Fusion AUC: {summary.get('fusion_auc', 0):.4f}, AP: {summary.get('fusion_ap', 0):.4f}")
    print(f"  Fusion Precision: {summary.get('fusion_precision', 0):.4f}, Recall: {summary.get('fusion_recall', 0):.4f}, F1: {summary.get('fusion_f1', 0):.4f}")
    
    print("\nNode Prediction Performance:")
    for node_type in node_types:
        print(f"  {node_type.upper()} Prediction:")
        print(f"    MRR: {summary.get(f'{node_type}_mrr', 0):.4f}")
        print(f"    Hit@1: {summary.get(f'{node_type}_hit@1', 0):.4f}")
        print(f"    Hit@3: {summary.get(f'{node_type}_hit@3', 0):.4f}")
        print(f"    Hit@10: {summary.get(f'{node_type}_hit@10', 0):.4f}")
    print("=" * 50)
    
    # Save detailed results
    with open("stage2_evaluation/detailed_results.json", 'w') as f:
        json.dump({
            'metrics': metrics,
            'summary': summary,
            'path_pred_metrics': path_pred_metrics
        }, f, indent=2, ensure_ascii=False)
    
    # Plot path score distribution
    all_path_scores = [score for pq in metrics['path_quality'] for score in [pq['avg_score']] if pq['avg_score'] > 0]
    if all_path_scores:
        plt.figure()
        plt.hist(all_path_scores, bins=20, alpha=0.7)
        plt.xlabel('Path Score')
        plt.ylabel('Count')
        plt.title('Path Score Distribution')
        plt.savefig("stage2_evaluation/path_score_distribution.png")
        plt.close()
    
    # Plot score comparison
    plt.figure(figsize=(10, 6))
    plt.scatter(metrics['stage1_scores'], metrics['fusion_scores'], alpha=0.6)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Stage1 Score')
    plt.ylabel('Fusion Score')
    plt.title('Stage1 vs Fusion Scores')
    plt.grid(True)
    plt.savefig("stage2_evaluation/score_comparison.png")
    plt.close()
    
    total_time = time.time() - start_time
    print(f"\nEvaluation completed! Total time: {total_time:.2f} seconds")
    print(f"Detailed results saved to stage2_evaluation/ directory")
    print("=" * 50)

if __name__ == "__main__":
    evaluate_stage2()