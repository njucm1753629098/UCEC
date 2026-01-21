import torch
import sys
import time
from data_loader import load_data
from model import Stage1Model, Stage2Model
from utils import advanced_fusion, find_paths, is_original_path
import pickle
from tqdm import tqdm
import numpy as np
import json
import os
import pandas as pd
from datetime import datetime

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

def predict(herb_name):
    print("=" * 50)
    print(f"Predicting effects for herb '{herb_name}'")
    print("=" * 50)
    start_time = time.time()
    
    # Create output directory
    output_dir = f"predictions/{herb_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data and models
    data, mappings = load_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Ensure graph data is on the correct device
    data = data.to(device)
    
    # Get herb index
    if herb_name not in mappings['herb']:
        print(f"Error: Herb '{herb_name}' not found")
        return
    
    herb_idx = mappings['herb'][herb_name]
    
    # Load Stage1 model
    stage1_model = Stage1Model(data, hidden_channels=128, out_channels=64, num_relations=RELATION_TYPES).to(device)
    stage1_model.load_state_dict(torch.load("best_stage1_model.pth"))
    stage1_model.eval()
    
    # Load Stage2 model
    stage2_model = Stage2Model(hidden_channels=64).to(device)
    stage2_model.load_state_dict(torch.load("best_stage2_model.pth"))
    stage2_model.eval()
    
    # Stage1 prediction for ALL symptoms
    print("Performing initial herb-symptom association prediction for ALL symptoms...")
    with torch.no_grad():
        x_dict = stage1_model(None, data.edge_index_dict)
        scores_stage1 = stage1_model.predict_herb_symptom(x_dict)
    
    # Pass node embeddings to Stage2 model
    stage2_model.x_dict = x_dict
    
    # Get all symptoms
    all_symptoms = list(range(len(mappings['symptom'])))
    print(f"Total symptoms to analyze: {len(all_symptoms)}")
    
    # Stage2 prediction for ALL symptoms
    print("Generating explanation chains for ALL symptoms...")
    results = []
    detailed_data = []
    
    # Process all symptoms
    for symptom_idx in tqdm(all_symptoms, desc="Generating predictions"):
        symptom_name = mappings['reverse_symptom'][symptom_idx]
        
        # Save Stage1 score
        stage1_score = scores_stage1[herb_idx, symptom_idx].item()
        
        # Find top paths (for every symptom)
        paths = find_paths(data, herb_idx, symptom_idx, stage2_model, mappings, k=5)
        path_scores = [score for _, score in paths] if paths else [0.0]
        paths_list = [path for path, _ in paths] if paths else []
        
        # Use advanced fusion algorithm
        final_score = advanced_fusion(
            stage1_score,
            path_scores,
            paths_list,
            data
        )
        
        # Save results
        result = {
            'symptom_idx': symptom_idx,
            'symptom_name': symptom_name,
            'stage1_score': stage1_score,
            'final_score': final_score,
            'path_scores': path_scores,
            'paths': paths
        }
        results.append(result)
        
        # Save detailed data for CSV
        if paths:
            for path_idx, (path, path_score) in enumerate(paths):
                detailed_data.append({
                    'herb': herb_name,
                    'symptom': symptom_name,
                    'stage1_score': stage1_score,
                    'path_score': path_score,
                    'final_score': final_score,
                    'path_index': path_idx + 1,
                    'target': mappings['reverse_target'][path['target']],
                    'pathway': mappings['reverse_pathway'][path['pathway']],
                    'disease': mappings['reverse_disease'][path['disease']],
                    'is_original': is_original_path(path, data)
                })
        else:
            detailed_data.append({
                'herb': herb_name,
                'symptom': symptom_name,
                'stage1_score': stage1_score,
                'path_score': 0.0,
                'final_score': final_score,
                'path_index': 0,
                'target': "N/A",
                'pathway': "N/A",
                'disease': "N/A",
                'is_original': False
            })
    
    # Sort by final score
    results.sort(key=lambda x: x['final_score'], reverse=True)
    
    # Output top 10 results
    print("\n" + "=" * 50)
    print(f"Top 10 prediction results for herb '{herb_name}':")
    print("=" * 50)
    
    for i, res in enumerate(results[:10]):
        print(f"\n{i + 1}. Symptom: {res['symptom_name']} - Stage1: {res['stage1_score']:.4f} - Final: {res['final_score']:.4f}")
        if res['paths']:
            for j, (path, path_score) in enumerate(res['paths']):
                original = "Original" if is_original_path(path, data) else "Predicted"
                print(f"   Path {j + 1} ({original}, Score: {path_score:.4f}):")
                print(f"     Herb: {mappings['reverse_herb'][path['herb']]}")
                print(f"     → Target: {mappings['reverse_target'][path['target']]}")
                print(f"     → Pathway: {mappings['reverse_pathway'][path['pathway']]}")
                print(f"     → Disease: {mappings['reverse_disease'][path['disease']]}")
                print(f"     → Symptom: {mappings['reverse_symptom'][path['symptom']]}")
        else:
            print("   No paths found")

    # Save detailed results to CSV
    df = pd.DataFrame(detailed_data)
    csv_path = os.path.join(output_dir, f"{herb_name}_all_predictions.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\nDetailed prediction data for all symptoms saved to: {csv_path}")
    
    # Save JSON results
    json_path = os.path.join(output_dir, f"{herb_name}_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'herb': herb_name,
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2, ensure_ascii=False)
    print(f"Full prediction results saved to: {json_path}")

    print("\n" + "=" * 50)
    total_time = time.time() - start_time
    print(f"Prediction completed! Total time: {total_time:.2f} seconds")
    print("=" * 50)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <chinese_herb_name>")
        sys.exit(1)
    predict(sys.argv[1])