# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies

```bash
pip install numpy pandas scikit-learn torch matplotlib xgboost lightgbm
```

### Step 2: Run Your First Model

```bash
# Try a baseline model first
python BaselinePerformance2_method.py

# Or try a classic model (usually better)
python Classic_model_3_method.py  # XGBoost
```

### Step 3: Check Results

- Look at the console output for **ROC AUC scores**
- Check `my_submission.csv` for predictions
- Compare models and pick the best one!

## 📊 Model Quick Reference

| Model              | File                             | Expected ROC AUC | Speed          |
| ------------------ | -------------------------------- | ---------------- | -------------- |
| Logistic (sklearn) | `BaselinePerformance2_method.py` | 0.80-0.85        | ⚡ Fast        |
| Logistic (PyTorch) | `BaselinePerformance1_method.py` | 0.80-0.85        | ⚡ Fast        |
| Random Forest      | `Classic_model_1_method.py`      | 0.83-0.87        | 🐢 Medium      |
| Gradient Boosting  | `Classic_model_2_method.py`      | 0.84-0.88        | 🐢 Medium      |
| **XGBoost**        | `Classic_model_3_method.py`      | **0.85-0.90**    | ⚡ Fast        |
| **LightGBM**       | `Classic_model_4_method.py`      | **0.85-0.90**    | ⚡⚡ Very Fast |
| Neural Network     | `ImprovedNN_modelMethod.py`      | 0.82-0.88        | 🐌 Slow        |

**Recommendation**: Start with **XGBoost** or **LightGBM** - they usually perform best!

## 🎯 Recommended Workflow

1. **Baseline**: Run `BaselinePerformance2_method.py` to establish baseline
2. **Best Models**: Run XGBoost and LightGBM
3. **Compare**: Note validation ROC AUC scores
4. **Tune**: Adjust hyperparameters of best model
5. **Submit**: Use best model's `my_submission.csv`

## 📝 Track Your Results

After running a model, record the results:

```python
from results_tracker import ResultsTracker

tracker = ResultsTracker()
tracker.add_result(
    model_name="XGBoost_v1",
    model_type="Classic",
    roc_auc_train=0.8756,  # From console output
    roc_auc_val=0.8612,    # From console output
    hyperparameters={'n_estimators': 200, 'learning_rate': 0.1},
    notes="First attempt",
    submission_file="my_submission.csv"
)

# View all results
tracker.print_summary()
```

## 🔧 Tune Hyperparameters

### XGBoost (usually best)

Edit `Classic_model_3_method.py`:

```python
n_estimators = 300  # Increase for better performance (slower)
learning_rate = 0.05  # Lower = better but slower
max_depth = 8  # Deeper = more complex
```

### Neural Network

Edit `ImprovedNN_modelMethod.py`:

```python
hidden_layers = [256, 128, 64]  # Deeper network
dropout_rate = 0.4  # More regularization
learning_rate = 5e-5  # Lower learning rate
```

## ⚠️ Common Issues

**Issue**: `ModuleNotFoundError: No module named 'xgboost'`  
**Fix**: `pip install xgboost`

**Issue**: `ModuleNotFoundError: No module named 'lightgbm'`  
**Fix**: `pip install lightgbm`

**Issue**: Model takes too long  
**Fix**: Reduce `n_estimators` or `n_epochs` in model file

## 📤 Submission Format

Your `my_submission.csv` should look like:

```csv
id,loan_paid_back
593994,0.5234
593995,0.7123
593996,0.3456
...
```

- Header row required
- `loan_paid_back` must be probabilities (0.0 to 1.0)
- One row per test sample

## 🏆 Tips for Best Performance

1. **Start with tree models** (XGBoost/LightGBM) - they're usually best
2. **Tune hyperparameters** of the best model
3. **Try ensemble** - average predictions from top 2-3 models
4. **Feature engineering** - create new features if needed
5. **Cross-validation** - use K-fold for more robust evaluation

## 📚 More Information

- See `README.md` for full documentation
- See `PROJECT_SUMMARY.md` for what's implemented
- See `example_track_results.py` for tracking examples

## 🎓 Competition Checklist

- [ ] Run baseline models
- [ ] Run classic models (especially XGBoost/LightGBM)
- [ ] Run neural network
- [ ] Record all ROC AUC scores
- [ ] Tune best model's hyperparameters
- [ ] Generate final submission
- [ ] Verify submission format
- [ ] Submit!

Good luck! 🍀
