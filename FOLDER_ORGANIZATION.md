# Folder Organization

## Structure

```
Project_2_AI/
├── submissions/                          # All submission CSV files
│   ├── my_submission.csv                 # Main/latest submission
│   ├── my_submission_advanced.csv        # Advanced ensemble
│   ├── my_submission_catboost.csv        # CatBoost ensemble
│   ├── my_submission_combined.csv        # Combined ensemble
│   ├── my_submission_weighted_*.csv      # Weighted ensemble variants
│   └── README.md                         # Submissions folder info
│
├── instructions_and_information/         # All documentation
│   ├── README.md                         # Documentation index
│   ├── EXPERIMENT_LOG.md                 # Complete experiment log
│   ├── RESULTS_ANALYSIS.md               # Results analysis
│   ├── QUICK_START.md                    # Quick start guide
│   ├── PUSH_FOR_HIGHER.md                # Advanced techniques
│   └── ... (other .md files)
│
├── Datasets/                             # Data files
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── Examples_Class/                       # Example code
│
└── [Python scripts]                     # All model scripts in root
    ├── ensemble_advanced.py
    ├── ensemble_catboost.py
    ├── feature_engineering_advanced.py
    └── ...
```

## Key Changes

### ✅ Submissions Folder
- **Location**: `submissions/`
- **Contains**: All CSV submission files
- **Naming**: `my_submission_<method>.csv`
- **Updated Scripts**: All scripts now save to `submissions/` folder

### ✅ Documentation Folder
- **Location**: `instructions_and_information/`
- **Contains**: All markdown documentation files
- **Includes**: Guides, logs, analysis, instructions

## Updated Scripts

The following scripts have been updated to save submissions to `submissions/`:
- ✅ `ensemble_advanced.py`
- ✅ `ensemble_catboost.py`
- ✅ `ensemble_method.py`
- ✅ `ensemble_weighted_advanced.py`
- ✅ `feature_engineering_advanced.py`
- ✅ `run_both_ensembles.py`

## Usage

### Running Scripts
Scripts still run from the root folder:
```bash
python ensemble_advanced.py
```

### Finding Submissions
All submissions are now in:
```bash
ls submissions/
```

### Reading Documentation
All docs are in:
```bash
ls instructions_and_information/
```

## Benefits

1. **Clean Organization**: Submissions and docs are separated
2. **Easy to Find**: Clear folder structure
3. **Future-Proof**: New submissions automatically go to `submissions/`
4. **Documentation**: README files in each folder explain contents

## Notes

- Old scripts that haven't been updated will still save to root (but you can move them)
- The main `my_submission.csv` is always in `submissions/` folder
- All future submissions will follow this organization

