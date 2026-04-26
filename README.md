# UCEC: Uncertainty-Aware Evidence-Chain Reasoning for TCM Repositioning

This repository provides a runnable implementation of the UCEC model for traditional Chinese medicine repositioning. The code trains a relation-aware graph neural network, retrieves mechanistic evidence chains, refines herb-disease prediction scores, estimates prediction uncertainty, and exports readable prediction tables.


## Repository Structure

```text
ucec/
  data.py
  graph.py
  preprocess.py
  proxy.py
  schema.py
  stage2.py
  training.py
  utils.py
  models/
  scripts/
data/
outputs/
requirements.txt
README.md
```

## Main Scripts

- `ucec/scripts/prepare_splits.py`: builds graph splits and derived pathway-disease edges.
- `ucec/scripts/train_stage1.py`: trains the stage-1 R-GCN graph model.
- `ucec/scripts/run_stage2_ucec.py`: trains the stage-2 evidence-chain model, estimates uncertainty, and exports prediction outputs.
- `ucec/scripts/export_readable_predictions.py`: converts model outputs into readable herb-disease predictions with evidence chains.
- `ucec/scripts/screen_case_candidates_final.py`: screens representative herb-disease cases from exported predictions.

## Environment

Python 3.9 or later is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

For GPU training, install a CUDA-enabled PyTorch build that matches your local CUDA version before running the pipeline. Check GPU availability with:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Run the Pipeline

Open a terminal in the repository root and set `PYTHONPATH`.

PowerShell:

```powershell
$env:PYTHONPATH = "$PWD"
```

Bash:

```bash
export PYTHONPATH="$PWD"
```

### 1. Prepare Graph Splits

```bash
python ucec/scripts/prepare_splits.py \
  --data_dir data \
  --out_dir outputs/runs \
  --seeds 1 \
  --auto_build_inputs
```

### 2. Train Stage 1 R-GCN

Use `--device cuda` if GPU-enabled PyTorch is available. Otherwise use `--device cpu`.

```bash
python ucec/scripts/train_stage1.py \
  --run_dir outputs/runs/seed_1 \
  --device cuda \
  --max_epochs 200 \
  --patience 20 \
  --min_delta 0.0001 \
  --eval_every 1 \
  --steps_per_epoch 50 \
  --batch_pos 512 \
  --neg_per_pos 1 \
  --dim 128 \
  --dropout 0.2 \
  --lr 0.002 \
  --weight_decay 0.0 \
  --print_every 1
```

### 3. Train Stage 2 and Export Predictions

```bash
python ucec/scripts/run_stage2_ucec.py \
  --run_dir outputs/runs/seed_1 \
  --data_dir data \
  --device cuda \
  --n_herbs 500 \
  --pos_per_herb 4 \
  --neg_per_herb 16 \
  --use_ppi_hop \
  --mc_samples 16 \
  --retrieval_budget 100 \
  --aggregation_budget 30 \
  --stage2_max_epochs 200 \
  --stage2_patience 20 \
  --stage2_min_delta 0.0001 \
  --stage2_monitor_fraction 0.25 \
  --stage2_train_batch_size 64 \
  --stage2_train_lr 0.001 \
  --stage2_print_every 1 \
  --export_full_predictions \
  --full_shortlist_per_herb 100 \
  --full_topk_per_herb 30 \
  --full_herb_limit 0 \
  --full_explanation_pairs 300
```

### 4. Export Readable Predictions

```bash
python ucec/scripts/export_readable_predictions.py \
  --run_dir outputs/runs/seed_1 \
  --device cuda
```
