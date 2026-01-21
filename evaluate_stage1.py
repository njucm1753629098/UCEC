import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from data_loader import load_data
from model import Stage1Model
from utils import generate_positive_samples, generate_negative_samples, split_dataset
import os
import sys
import pandas as pd
from tqdm import tqdm

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

def evaluate_saved_model(model_path, test_size=0.2, random_seed=42):
    """Evaluate a saved model on the test set - including visualization"""
    print("="*50)
    print("Start evaluating the saved model")
    print("="*50)
    
    # Set random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Load data
    data, mappings = load_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = Stage1Model(data, hidden_channels=128, out_channels=64, num_relations=RELATION_TYPES).to(device)
    data = data.to(device)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path))
    print(f"Model loaded: {model_path}")
    
    # Generate samples
    positive_labels = generate_positive_samples(data, mappings)
    num_herbs = len(mappings['herb'])
    num_symptoms = len(mappings['symptom'])
    
    # Generate negative samples
    neg_edge_index = generate_negative_samples(
        positive_labels, 
        num_herbs, 
        num_symptoms, 
        neg_ratio=1.0,
        augment_factor=1
    )
    
    # Split test set
    _, _, test_pos, test_neg = split_dataset(
        positive_labels, 
        neg_edge_index, 
        test_size=test_size, 
        random_seed=random_seed
    )
    
    # Move to device
    test_pos = test_pos.to(device)
    test_neg = test_neg.to(device)
    
    # Evaluate model
    metrics, all_preds, all_labels = evaluate_model(model, test_pos, test_neg)
    
    # Print evaluation results
    print("\n" + "="*50)
    print("Model Evaluation Results:")
    print(f"  Accuracy: {metrics['acc']:.4f}")
    print(f"  Positive Accuracy: {metrics['pos_acc']:.4f}")
    print(f"  Negative Accuracy: {metrics['neg_acc']:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print("="*50)
    
    # Create visualization directory
    output_dir = "evaluation_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot ROC curve
    plot_roc_curve(all_labels, all_preds, metrics['auc'], output_dir)
    
    # Plot PR curve
    plot_pr_curve(all_labels, all_preds, metrics['precision'], metrics['recall'], 
                  metrics['f1'], output_dir)
    
    # Analyze training log (if exists)
    log_file = find_training_log(model_path)
    if log_file:
        print(f"Training log file found: {log_file}")
        plot_training_curves(log_file, output_dir)
    else:
        print("No training log file found, skipping training curve plotting")
    
    print(f"\nAll visualization plots have been saved to: {output_dir}")

def evaluate_model(model, pos_edge_index, neg_edge_index):
    """Evaluate model performance - return predictions for visualization"""
    model.eval()
    with torch.no_grad():
        # Get all node embeddings
        x_dict = model(None, model.data.edge_index_dict)
        
        # Predict scores
        scores = model.predict_herb_symptom(x_dict)
        
        # Calculate positive predictions
        pos_preds = scores[pos_edge_index[0], pos_edge_index[1]]
        
        # Calculate negative predictions
        neg_preds = scores[neg_edge_index[0], neg_edge_index[1]]
        
        # Calculate accuracy
        pos_acc = (pos_preds > 0.5).float().mean().item()
        neg_acc = (neg_preds < 0.5).float().mean().item()
        total_acc = (pos_acc * len(pos_preds) + neg_acc * len(neg_preds)) / (len(pos_preds) + len(neg_preds))
        
        # Prepare data for AUC calculation
        all_preds = torch.cat([pos_preds, neg_preds]).cpu().numpy()
        all_labels = torch.cat([
            torch.ones_like(pos_preds), 
            torch.zeros_like(neg_preds)
        ]).cpu().numpy()
        
        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(all_labels, all_preds)
        
        # Calculate F1 score
        predictions = (all_preds > 0.5).astype(int)
        true_pos = ((predictions == 1) & (all_labels == 1)).sum()
        false_pos = ((predictions == 1) & (all_labels == 0)).sum()
        false_neg = ((predictions == 0) & (all_labels == 1)).sum()
        
        precision = true_pos / (true_pos + false_pos + 1e-10)
        recall = true_pos / (true_pos + false_neg + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # Return evaluation metrics and raw predictions
        metrics = {
            'acc': total_acc,
            'pos_acc': pos_acc,
            'neg_acc': neg_acc,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return metrics, all_preds, all_labels

def plot_roc_curve(labels, preds, auc_score, output_dir):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    
    # Mark the best threshold
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label=f'Best Threshold={gmeans[ix]:.2f}')
    
    # Save image
    output_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to: {output_path}")

def plot_pr_curve(labels, preds, precision, recall, f1, output_dir):
    """Plot PR curve"""
    precision_curve, recall_curve, _ = precision_recall_curve(labels, preds)
    average_precision = average_precision_score(labels, preds)
    
    plt.figure()
    plt.plot(recall_curve, precision_curve, color='blue', lw=2, 
             label=f'PR curve (AP = {average_precision:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Precision-Recall Curve (PR)')
    plt.legend(loc="upper right")
    
    # Mark F1 score
    plt.scatter(recall, precision, marker='o', color='red', 
                label=f'F1 Score={f1:.4f}\nPrecision={precision:.4f}\nRecall={recall:.4f}')
    
    # Save image
    output_path = os.path.join(output_dir, "pr_curve.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"PR curve saved to: {output_path}")

def find_training_log(model_path):
    """Find the training log related to the model - Fix empty directory issue"""
    # Try to infer the log file name from the model path
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path)
    
    # Handle empty directory path
    if not model_dir:
        model_dir = "."
    
    # If it is best_model, find the most recent log
    if "best" in model_name:
        # Ensure directory exists
        if os.path.exists(model_dir):
            log_files = [f for f in os.listdir(model_dir) if f.startswith("stage1_training_progress")]
            if log_files:
                return os.path.join(model_dir, sorted(log_files)[-1])
    
    # Try to match a specific pattern
    timestamp = model_name.split("_")[-1].split(".")[0]
    if len(timestamp) == 14:  # Similar to 20250620_002727
        log_file = f"stage1_training_progress_{timestamp}.csv"
        log_path = os.path.join(model_dir, log_file)
        if os.path.exists(log_path):
            return log_path
    
    return None

def plot_training_curves(log_file, output_dir):
    """Plot training curves from training log"""
    # Read log data
    df = pd.read_csv(log_file)
    
    # 1. Loss curve
    plt.figure()
    plt.plot(df['epoch'], df['loss'], label='Total Loss', color='blue')
    plt.plot(df['epoch'], df['pos_loss'], label='Positive Loss', color='green', linestyle='--')
    plt.plot(df['epoch'], df['neg_loss'], label='Negative Loss', color='red', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    
    # Save image
    output_path = os.path.join(output_dir, "training_loss.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Training loss curve saved to: {output_path}")
    
    # 2. Accuracy curve
    plt.figure()
    plt.plot(df['epoch'], df['acc'], label='Total Accuracy', color='blue')
    plt.plot(df['epoch'], df['pos_acc'], label='Positive Accuracy', color='green', linestyle='--')
    plt.plot(df['epoch'], df['neg_acc'], label='Negative Accuracy', color='red', linestyle='--')
    plt.plot(df['epoch'], df['test_acc'], label='Test Accuracy', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Curve')
    plt.legend()
    plt.grid(True)
    
    # Save image
    output_path = os.path.join(output_dir, "training_accuracy.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Training accuracy curve saved to: {output_path}")
    
    # 3. Advanced metrics curve
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(df['epoch'], df['test_auc'], label='Test AUC', color='blue')
    plt.ylabel('AUC')
    plt.title('Test AUC Curve')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(df['epoch'], df['test_f1'], label='Test F1 Score', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Test F1 Score Curve')
    plt.grid(True)
    
    # Save image
    output_path = os.path.join(output_dir, "advanced_metrics.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Advanced metrics curve saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate_model.py <model_path>")
        sys.exit(1)
    evaluate_saved_model(sys.argv[1])