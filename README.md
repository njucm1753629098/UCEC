# UCEC 

## 0) Install
Create a fresh environment and install:
```bash
pip install -r requirements.txt
```

## 1) Run (per seed)
### A) Prepare split-aware inputs 
```bash
python scripts/01_prepare_splits.py --data_dir /path/to/data --out_dir runs
```

### B) Train Stage-1 models (R-GCN + baselines) and evaluate PD / IP link prediction
```bash
python scripts/02_train_stage1.py --run_dir runs/seed_1
```

### C) Run Stage-2 UCEC (evidence chains) + proxy benchmark + calibration + deletion tests
```bash
python scripts/03_run_stage2_ucec.py --run_dir runs/seed_1
```
