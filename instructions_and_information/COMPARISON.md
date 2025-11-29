# Ensemble Methods Comparison

## Key Differences

### `ensemble_advanced.py`

- **Models**: Random Forest, Gradient Boosting, XGBoost, LightGBM, Extra Trees (5 models)
- **Features**:
  - Feature engineering (interaction features)
  - Income-to-loan ratio
  - Debt-score interactions
- **Methods**: Simple average, weighted average, optimized weights, stacking
- **Focus**: More models + feature engineering

### `ensemble_catboost.py`

- **Models**: CatBoost, XGBoost, LightGBM, Gradient Boosting (4 models)
- **Features**: Standard preprocessing (no feature engineering)
- **Methods**: Simple average, weighted average, optimized weights
- **Focus**: CatBoost (often best single model)

## Why Run Both?

1. **Different Models**:

   - Advanced has Extra Trees (different algorithm)
   - CatBoost has CatBoost (often outperforms XGBoost/LightGBM)

2. **Different Approaches**:

   - Advanced uses feature engineering
   - CatBoost relies on native categorical handling

3. **Complementary**: They may capture different patterns in the data

4. **Combining**: Averaging both can reduce variance and improve generalization

## How to Use

### Option 1: Run Both Separately

```bash
# Run advanced ensemble
python ensemble_advanced.py
# Check my_submission.csv, save if good

# Run CatBoost ensemble
pip install catboost  # if not installed
python ensemble_catboost.py
# Check my_submission.csv, save if good
```

### Option 2: Run Both Automatically (Recommended)

```bash
python run_both_ensembles.py
```

This will:

1. Run both ensemble methods
2. Save individual results to:
   - `my_submission_advanced.csv`
   - `my_submission_catboost.csv`
3. Create combined ensemble:
   - `my_submission_combined.csv` (RECOMMENDED)
   - Updates `my_submission.csv`

## Expected Results

| Method            | Expected Score  | Notes                          |
| ----------------- | --------------- | ------------------------------ |
| Advanced Ensemble | 0.921-0.923     | Feature engineering + 5 models |
| CatBoost Ensemble | 0.921-0.923     | CatBoost often best            |
| **Combined**      | **0.922-0.925** | **Best of both worlds**        |

## Recommendation

**Use the combined ensemble** (`my_submission_combined.csv`):

- Combines strengths of both approaches
- Reduces variance (more stable)
- Usually performs better than either alone
- Expected improvement: +0.002 to +0.005 over 0.92045
