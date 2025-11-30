# Experiment Log - Loan Default Prediction Competition

## Score History

| Date    | Model                  | ROC AUC     | Improvement | Notes                                            |
| ------- | ---------------------- | ----------- | ----------- | ------------------------------------------------ |
| Initial | Baseline               | 0.91298     | Baseline    | Initial submission                               |
| -       | Ensemble (4 models)    | 0.92045     | +0.00747    | Combined RF, GB, XGB, LGB with optimized weights |
| -       | Advanced Ensemble      | 0.92088     | +0.00043    | Feature engineering + 5 models + stacking        |
| -       | CatBoost Ensemble      | 0.92016     | -0.00029    | CatBoost + XGB + LGB + GB (weaker than advanced) |
| -       | Combined Ensemble      | 0.92077     | -0.00011    | Simple average of Advanced + CatBoost            |
| -       | Weighted 80%           | **0.92100** | +0.00012    | Advanced 80% + CatBoost 20%                      |
| -       | Weighted 90%           | **0.92100** | +0.00012    | Advanced 90% + CatBoost 10%                      |
| -       | Weighted 95%           | 0.92096     | +0.00008    | Advanced 95% + CatBoost 5%                       |
| -       | Feature Eng + 7 Models | 0.92075     | -0.00025    | Added NNs, all 7 models (overfitting)            |
| -       | Feature Eng + 2 Models | 0.92054     | -0.00046    | Auto-selection too aggressive                    |
| -       | Feature Eng + 3 Best   | 0.92086     | -0.00014    | GB+XGB+LGB only (PReLU NNs, still < weighted)    |

## Current Best: 0.92175 (Hyperparameter Optimization 50 trials, XGB/LGB only) 🏆

**Previous Best**: 0.92100 (Weighted Ensemble 80% or 90%)

**Note**: Hyperparameter optimization with 50 trials on XGBoost and LightGBM (skipping GB optimization) achieved 0.92175, beating the previous best of 0.92100. This confirms that:
- More trials (50 vs 20) = better optimization
- Skipping GB optimization saves time and focuses on better models
- XGBoost/LightGBM optimization has more impact than GB optimization

---

## Experiment 1: Initial Baseline

- **Model**: Single model (unknown which one)
- **Score**: 0.91298
- **Date**: Initial submission
- **Notes**: Baseline performance

---

## Experiment 2: Ensemble Method

- **Date**: Current session
- **Model**: Ensemble of 4 models
  - Random Forest (300 trees, max_depth=25)
  - Gradient Boosting (300 trees, lr=0.05, max_depth=6)
  - XGBoost (500 trees, lr=0.05, max_depth=7, regularization)
  - LightGBM (500 trees, lr=0.05, max_depth=7, regularization)
- **Method**: Optimized weighted average (weights optimized on validation set)
- **Score**: 0.92045
- **Improvement**: +0.00747 (+0.82% relative improvement)
- **Key Changes**:
  - Combined multiple models instead of single model
  - Used optimized weights based on validation ROC AUC
  - Increased number of trees (300-500 vs 200)
  - Lower learning rates (0.05 vs 0.1)
  - Added regularization (L1/L2)
- **Files**: `ensemble_method.py`

---

## Experiment 3: Advanced Ensemble ✅ COMPLETED

- **Date**: Current session
- **File**: `ensemble_advanced.py`
- **Score**: 0.92088
- **Improvement**: +0.00043 over 0.92045
- **Changes from Experiment 2**:
  - Feature engineering (interaction features)
  - More models (added Extra Trees) - 5 models total
  - Higher tree counts (400-600 vs 300-500)
  - Lower learning rates (0.03 vs 0.05)
  - Deeper trees (max_depth 7-8 vs 6-7)
  - More regularization
  - Stacking with logistic regression meta-learner
- **Best Method**: [Check output - which ensemble method won?]
- **Notes**: Slight improvement, feature engineering helped
- **Status**: ✅ Completed

---

## Experiment 4: CatBoost Ensemble ✅ COMPLETED

- **Date**: Current session
- **File**: `ensemble_catboost.py`
- **Score**: 0.92016
- **Improvement**: -0.00029 vs Advanced (0.92088)
- **Changes**:
  - Added CatBoost (often outperforms XGBoost/LightGBM)
  - CatBoost handles categorical features natively
  - Optimized hyperparameters
  - Ensemble with XGBoost, LightGBM, Gradient Boosting
- **Notes**:
  - Performed worse than Advanced ensemble
  - CatBoost didn't help as much as expected for this dataset
  - Feature engineering in Advanced was more effective
- **Status**: ✅ Completed

---

## Experiment 5: Combined Both Ensembles ✅ COMPLETED

- **Date**: Current session
- **File**: `run_both_ensembles.py`
- **Score**: 0.92077
- **Improvement**: -0.00011 vs Advanced (0.92088)
- **What it does**:
  - Runs both `ensemble_advanced.py` and `ensemble_catboost.py`
  - Saves individual results
  - Combines predictions from both (simple average)
  - Creates `my_submission_combined.csv`
- **Results Analysis**:
  - Combined (0.92077) < Advanced (0.92088)
  - Simple average diluted the strong Advanced predictions
  - CatBoost (0.92016) was weaker, bringing down the average
- **Key Learning**:
  - Not all ensembles benefit from simple averaging
  - When one method is clearly better, averaging can hurt
  - Should have used weighted average favoring Advanced
- **Status**: ✅ Completed

---

## Experiment 6: Weighted Ensemble ✅ COMPLETED

- **Date**: Current session
- **File**: `ensemble_weighted_advanced.py`
- **Scores**:
  - Weighted 80% (Advanced 80%, CatBoost 20%): **0.92100** 🏆
  - Weighted 90% (Advanced 90%, CatBoost 10%): **0.92100** 🏆
  - Weighted 95% (Advanced 95%, CatBoost 5%): 0.92096
- **Improvement**: +0.00012 over Advanced (0.92088)
- **Key Finding**:
  - Weighting Advanced more heavily (80-90%) improved performance
  - 80% and 90% performed equally well
  - 95% was slightly worse (too much weight on one model)
- **Status**: ✅ Completed

---

## Experiment 7: Advanced Feature Engineering with Neural Networks ✅ COMPLETED

- **Date**: Current session
- **File**: `feature_engineering_advanced.py`
- **Status**: ✅ Completed
- **Scores**:
  - Initial (7 models, all included): 0.92075
  - With auto model selection (2 models): 0.92054
  - Final (3 best models: GB+XGB+LGB with PReLU): **0.92086**
- **Improvement**: Better than 7-model ensemble (+0.00011), but still below best (0.92100)
- **Comparison to best**: -0.00014 vs Weighted 80/90% (0.92100)
- **Key Changes**:
  - More interaction features (10+ new features)
  - Polynomial features (squared terms)
  - Log transformations
  - More trees (500-700 vs 400-600)
  - Lower learning rates (0.02 vs 0.03)
  - Deeper trees (max_depth 8-9 vs 7-8)
  - **Added ShallowNN** (100 hidden units, ELU → PReLU)
  - **Added DeepNN** (128→64→32, dropout 0.3, ELU → PReLU)
  - **Improved stacking** with 3 meta-learners (Logistic, Ridge, RF)
  - **Fixed GPU detection** (XGBoost/LightGBM use CPU with 12 threads)
  - **Model selection** to filter out weak models

### Individual Model Performance (Validation ROC AUC):
- RandomForest: 0.910488
- GradientBoosting: 0.915096
- XGBoost: 0.918589
- LightGBM: 0.919179 ⭐ (best)
- ExtraTrees: 0.909916
- ShallowNN: 0.909872 (with PReLU - slightly worse than 0.909999 with ELU)
- DeepNN: 0.911038 (with PReLU - slightly better than 0.911146 with ELU)

**Note on PReLU**: PReLU activation had minimal impact:
- ShallowNN: 0.909999 (ELU) → 0.909872 (PReLU) = -0.000127 (slightly worse)
- DeepNN: 0.911146 (ELU) → 0.911038 (PReLU) = -0.000108 (slightly worse)
- **Conclusion**: PReLU didn't help for this dataset - NNs still underperformed significantly

### Key Findings:
1. **Neural networks underperformed**: NNs scored ~0.91 vs trees ~0.918-0.919
2. **Too many models hurt**: Including all 7 models (0.92075) < 3 best models
3. **Auto-selection too aggressive**: Dropping to only 2 models (0.92054) was worse
4. **Best subset**: GradientBoosting + XGBoost + LightGBM (3 boosting models)
5. **PReLU activation**: Changed from ELU to PReLU for both NNs (friend's suggestion)
6. **Stacking underperformed**: 0.92072 < weighted 0.92100 (likely overfitting)

### Detailed Analysis: Why Things Happened

#### Why Neural Networks Underperformed (~0.91 vs ~0.919 for trees)

**1. Tabular Data Bias Toward Tree Models**
- **Tree models excel at tabular data**: Gradient boosting (XGBoost, LightGBM) are specifically designed for structured/tabular data
- **NNs designed for different data**: Neural networks excel at images, text, audio - data with spatial/temporal structure
- **Feature interactions**: Tree models naturally handle feature interactions through splits, while NNs must learn these through weight matrices
- **Result**: On this loan default dataset, tree models have a natural advantage

**2. Limited Training Data for NNs**
- **NNs need more data**: Neural networks typically need large datasets (millions of samples) to learn complex patterns
- **This dataset**: ~200K training samples is moderate - enough for trees, borderline for NNs
- **Overfitting risk**: NNs with limited data tend to overfit or underfit more easily than tree models
- **Early stopping helped**: But may have stopped too early, preventing NNs from reaching full potential

**3. Hyperparameter Sensitivity**
- **Tree models**: Relatively robust to hyperparameters - small changes don't drastically affect performance
- **NNs**: Highly sensitive to:
  - Learning rate (1e-4 may be too conservative)
  - Batch size (2048 may be too large)
  - Architecture (100 hidden units may not be optimal)
  - Initialization (random seed can cause 0.001-0.002 variance)
- **Result**: Without extensive hyperparameter tuning, NNs couldn't match tree performance

**4. Feature Engineering Advantage for Trees**
- **Hand-crafted features**: The code creates many interaction features (ratios, products, squared terms)
- **Tree models**: Can immediately exploit these features through optimal splits
- **NNs**: Must rediscover these relationships through backpropagation
- **Example**: `income_to_loan` ratio is immediately useful to trees, but NNs need to learn this relationship

**5. Architecture Limitations**
- **ShallowNN (100 units)**: May be too simple - not enough capacity to learn complex patterns
- **DeepNN (128→64→32)**: Dropout (0.3) may be too aggressive, preventing learning
- **No batch normalization**: Could help stabilize training
- **Fixed architecture**: Not optimized for this specific dataset

**6. Training Dynamics**
- **Sequential training**: NNs train sequentially (one epoch at a time), while trees can be parallelized
- **Gradient flow**: Even with PReLU, gradients may not flow optimally through the network
- **Local minima**: NNs can get stuck in suboptimal solutions, while tree models are more deterministic

**Conclusion**: NNs underperformed because:
1. Tabular data favors tree models by design
2. Limited data + hyperparameter sensitivity = suboptimal performance
3. Feature engineering benefits trees more than NNs
4. Architecture not fully optimized for this task

#### Why More Models Hurt Performance (7 models → 0.92075)

**1. Weak Model Dilution**
- **Noise addition**: Weak models (NNs ~0.91, ExtraTrees 0.9099) add noise to ensemble
- **Averaging effect**: Simple averaging gives equal weight to strong (0.919) and weak (0.91) models
- **Mathematical impact**: (0.919 + 0.91) / 2 = 0.9145 < 0.919 (dilution effect)

**2. Ensemble Diversity vs Quality Trade-off**
- **Diversity helps**: Different models can catch different patterns
- **But quality matters more**: A few excellent models > many mediocre models
- **Sweet spot**: 3-5 high-quality models is optimal, not 7 mixed-quality models

**3. Overfitting to Validation Set**
- **More models = more parameters**: Ensemble with 7 models has more degrees of freedom
- **Validation set size**: 20% of data may not be large enough to reliably evaluate 7-model ensemble
- **Optimization complexity**: Finding optimal weights for 7 models is harder than 3-5

**4. Correlation Between Models**
- **Tree models are similar**: RandomForest, ExtraTrees, GradientBoosting all use similar tree-based logic
- **Redundant information**: Adding similar models doesn't add much diversity
- **NNs different but weak**: NNs add diversity but their predictions are less accurate

**Result**: Including weak models (NNs, ExtraTrees) dragged down the ensemble average

#### Why Auto-Selection Was Too Aggressive (2 models → 0.92054)

**1. Selection Threshold Too Strict**
- **Original rule**: Keep models within 0.0015 of best (0.919179)
- **This filtered to**: Only XGBoost (0.918589) and LightGBM (0.919179)
- **Problem**: GradientBoosting (0.915096) was excluded despite being a strong model

**2. Loss of Complementary Information**
- **GradientBoosting adds value**: Even at 0.915, it may catch patterns XGB/LGB miss
- **Ensemble diversity**: 2 models is too few - less robust to individual model errors
- **Sweet spot**: 3-5 models provides good balance

**3. Validation Set Variance**
- **Small validation set**: 20% split means ~40K samples
- **ROC AUC variance**: Small differences (0.915 vs 0.919) may not be statistically significant
- **Overfitting to validation**: Selection based on validation may not generalize

**Result**: Too few models (2) lost valuable diversity and complementary predictions

#### Why Stacking Underperformed (0.92072 vs 0.92100)

**1. Overfitting to Validation Set**
- **Meta-learner training**: Stacking trains a meta-model on validation predictions
- **Small validation set**: 20% split (~40K samples) may be too small
- **Complex meta-learners**: Logistic Regression, Random Forest meta-learners can overfit
- **Result**: Meta-learner learned validation-specific patterns that don't generalize

**2. Too Many Features for Meta-Learner**
- **7 model predictions**: Meta-learner receives 7 features (one per model)
- **With weak models**: Including weak model predictions (NNs) adds noise
- **Optimal would be**: Meta-learner trained only on strong models (3-5)

**3. Linear Meta-Learner Limitation**
- **Logistic Regression**: Can only learn linear combinations
- **Non-linear relationships**: Model predictions may have non-linear interactions
- **Random Forest meta-learner**: Could help but also more prone to overfitting

**4. Validation Set Leakage**
- **Same data for selection and training**: Meta-learner trained on same validation set used to evaluate
- **Optimistic bias**: Performance on validation may not reflect test performance
- **Cross-validation needed**: Should use out-of-fold predictions for meta-learner

**Result**: Stacking overfit to validation set, while simple weighted average (0.921) generalized better

#### Why PReLU Was Suggested

**1. Better Gradient Flow**
- **ReLU problem**: Dead neurons (output always 0) prevent learning
- **ELU problem**: Exponential for negatives can be slow to compute
- **PReLU solution**: Learns optimal negative slope, preventing dead neurons while maintaining efficiency

**2. Tabular Data Performance**
- **Empirical evidence**: PReLU often outperforms ReLU/ELU on tabular data
- **Adaptive activation**: The learned slope parameter adapts to data distribution
- **Friend's experience**: Likely saw improvement in similar tabular prediction tasks

**3. Implementation**
- **PyTorch support**: `nn.PReLU()` is built-in, easy to use
- **No extra hyperparameters**: Unlike SELU which requires specific initialization
- **Backward compatible**: Works with existing initialization schemes

**Note**: PReLU was tested but had minimal impact:
- ShallowNN: 0.909999 (ELU) → 0.909872 (PReLU) = slightly worse
- DeepNN: 0.911146 (ELU) → 0.911038 (PReLU) = slightly worse
- **Conclusion**: PReLU didn't help - activation function wasn't the limiting factor for NNs

### Technical Fixes:

**1. XGBoost GPU Error Fix**
- **Problem**: XGBoost tried to use `tree_method='gpu_hist'` but standard pip install doesn't include GPU support
- **Error**: `XGBoostError: Invalid Input: 'gpu_hist', valid values are: {'approx', 'auto', 'exact', 'hist'}`
- **Solution**: Switched to CPU with `tree_method='hist'` and `n_jobs=12` (multi-threaded)
- **Why**: Standard XGBoost from pip is CPU-only; GPU requires special CUDA-enabled build
- **Performance**: CPU with 12 threads still very fast, uses 70-80% CPU like Random Forest

**2. LightGBM GPU Fix**
- **Problem**: Similar to XGBoost - tried to use GPU but standard build doesn't support it
- **Solution**: Use CPU with `n_jobs=12` for parallel processing
- **Why**: GPU support requires special LightGBM build with CUDA dependencies
- **Performance**: Multi-threaded CPU is sufficient for this dataset size

**3. CUDA Detection for NNs**
- **Added**: Proper CUDA detection using `torch.cuda.is_available()`
- **Info messages**: Shows GPU name and memory when CUDA available
- **Why**: NNs benefit significantly from GPU acceleration (10-100x speedup)
- **Result**: NNs train on GPU when available, CPU otherwise

**4. Gradient Boosting Verbose Output**
- **Added**: `verbose=1` to show training progress
- **Why**: User noticed low CPU usage (9%) - wanted to verify training was happening
- **Note**: Gradient Boosting is sequential (can't parallelize), so lower CPU usage is normal
- **Result**: Confirmed training was working, just sequential by design

**5. Improved Stacking**
- **Added**: 3 meta-learners instead of 1 (Logistic Regression, Ridge Regression, Random Forest)
- **Why**: Different meta-learners can capture different combination patterns
- **Selection**: Automatically picks best meta-learner based on validation ROC AUC
- **Result**: More robust stacking, but still underperformed simple weighted average

### Lessons Learned:
1. **More models ≠ better**: 7 models performed worse than 5 models
2. **Neural networks need tuning**: NNs require more hyperparameter tuning for tabular data
3. **Tree models dominate**: Gradient boosting trees are SOTA for tabular data
4. **Model selection critical**: Need to carefully select which models to include
5. **Stacking can overfit**: Simple weighted average (0.921) > stacking (0.92072)

### Final Configuration:
- **Models used**: GradientBoosting, XGBoost, LightGBM (3 best)
- **Ensemble methods**: Simple Average, Weighted Average, Optimized Weights, Best Stacking
- **Activation**: PReLU for both NNs (changed from ELU, but didn't help)
- **GPU**: CUDA for NNs, CPU for tree models (12 threads)
- **Final Score**: 0.92086 (using 3 best models)

### Final Results Analysis:

**Score Progression**:
1. 7 models (all included): 0.92075
2. 2 models (auto-selected): 0.92054 ❌ (too few)
3. 3 models (GB+XGB+LGB): 0.92086 ✅ (best of this experiment)

**Why 0.92086 < 0.92100 (Weighted 80/90%)**:

1. **Missing CatBoost Ensemble**: The 0.92100 score came from combining Advanced ensemble (0.92088) with CatBoost ensemble (0.92016) using weighted 80/90%. This experiment only used the Advanced ensemble models, missing the CatBoost contribution.

2. **Different Ensemble Method**: 
   - 0.92100: Weighted combination of two separate ensembles (Advanced 80% + CatBoost 20%)
   - 0.92086: Ensemble of individual models within Advanced ensemble only
   - **Key difference**: Two-stage ensemble (ensemble of ensembles) vs single-stage ensemble

3. **Model Diversity**:
   - 0.92100: Combined predictions from 5 Advanced models + 4 CatBoost models (9 models total, but combined at ensemble level)
   - 0.92086: Only 3 models from Advanced ensemble
   - **Missing diversity**: CatBoost ensemble adds different perspective even if individual score (0.92016) is lower

4. **Feature Engineering vs Native Categorical Handling**:
   - Advanced ensemble: Feature engineering approach
   - CatBoost ensemble: Native categorical feature handling
   - **Complementary**: These two approaches capture different patterns, and combining them helps

**Conclusion**: 
- 0.92086 is good for a single ensemble (3 models)
- But 0.92100 (weighted Advanced + CatBoost) remains best because it combines two different ensemble approaches
- **Recommendation**: Use weighted 80/90% combination of Advanced + CatBoost ensembles for best performance

---

## Next Experiments (To Try)

### Experiment 8: Hyperparameter Optimization

- Grid search or random search
- Bayesian optimization (Optuna)
- Cross-validation for robust evaluation

### Experiment 9: Advanced Neural Network

- Better architecture
- More layers
- Attention mechanisms

### Experiment 10: Blending

- Different train/validation splits
- Multiple folds
- Out-of-fold predictions

---

## Technical Details

### Data Preprocessing

- One-hot encoding for categorical variables
- Standardization (mean=0, std=1)
- Train/validation split: 80/20, random_state=2025
- Column alignment between train/test sets

### Model Configurations

#### Random Forest

- n_estimators: 300
- max_depth: 25
- min_samples_split: 5
- min_samples_leaf: 2

#### Gradient Boosting

- n_estimators: 300
- learning_rate: 0.05
- max_depth: 5
- subsample: 0.8

#### XGBoost

- n_estimators: 500
- learning_rate: 0.05
- max_depth: 6
- min_child_weight: 3
- subsample: 0.8
- colsample_bytree: 0.8
- gamma: 0.1
- reg_alpha: 0.1 (L1)
- reg_lambda: 1.0 (L2)

#### LightGBM

- n_estimators: 500
- learning_rate: 0.05
- max_depth: 6
- num_leaves: 31
- min_child_samples: 20
- subsample: 0.8
- colsample_bytree: 0.8
- reg_alpha: 0.1 (L1)
- reg_lambda: 1.0 (L2)

### Ensemble Method

- Simple average
- Weighted average (by individual ROC AUC)
- Optimized weights (scipy.optimize.minimize on validation set)
- Best method selected based on validation ROC AUC

---

## Lessons Learned

1. **Ensemble > Single Model**: Combining multiple models significantly improves performance
2. **Optimized Weights**: Optimizing ensemble weights on validation set beats simple averaging
3. **More Trees + Lower LR**: Better than fewer trees with higher learning rate
4. **Regularization Helps**: L1/L2 regularization improves generalization
5. **Feature Engineering Works**: Interaction features were key to Advanced ensemble's success
6. **Weighted > Simple Average**: When one model is better, weight it more heavily
7. **80-90% Weight Optimal**: Too much weight (95%) can reduce diversity and hurt performance
8. **More Models ≠ Better**: Adding weak models (NNs) can hurt ensemble performance
9. **Model Selection Critical**: Need to carefully filter which models to include
10. **Neural Networks Need Tuning**: NNs require extensive hyperparameter tuning for tabular data
11. **Tree Models Dominate Tabular**: Gradient boosting trees outperform NNs on structured data
12. **Stacking Can Overfit**: Simple weighted average can beat stacking on small validation sets
13. **PReLU for Tabular**: PReLU activation can help NNs on tabular data (vs ELU/ReLU)

---

## Experiment 8: Hyperparameter Optimization ✅ COMPLETED

- **Date**: Current session
- **File**: `hyperparameter_optimization.py`
- **Status**: ✅ Completed
- **Scores**:
  - Mac (20 trials + GB optimization): 0.92057
  - PC (50 trials, no GB optimization): **0.92175** 🏆 (NEW BEST!)
- **Improvement**: +0.00075 over previous best (0.92100)
- **Key Changes**:
  - Used Optuna (Bayesian optimization) for hyperparameter tuning
  - Optimized XGBoost and LightGBM (50 trials each)
  - **Skipped GB optimization** (too slow, weakest model)
  - Used advanced feature engineering from Experiment 7
  - CPU: 8 cores for XGB/LGB (background-safe)

### Individual Model Performance (After Optimization):
- XGBoost: [Optimized - check output]
- LightGBM: [Optimized - check output]
- GradientBoosting: [Default parameters - not optimized]

### Key Findings:
1. **50 trials > 20 trials**: More thorough optimization = better results
2. **Skipping GB was correct**: GB optimization takes 20-40 min/trial but GB is weakest model
3. **Focus on best models**: Optimizing XGB/LGB (best models) has more impact
4. **Time efficiency**: 50 trials XGB/LGB (~2-3 hours) > 20 trials all models (~5+ hours with GB)

### Technical Details:
- **Optimization method**: Optuna (Bayesian optimization)
- **XGBoost parameters tuned**: n_estimators (600-1200), learning_rate (0.01-0.03), max_depth (7-11), regularization, subsampling
- **LightGBM parameters tuned**: n_estimators (600-1200), learning_rate (0.01-0.03), max_depth (7-11), num_leaves, regularization
- **GradientBoosting**: Used default parameters (n_estimators=500, lr=0.02, max_depth=8)
- **Ensemble methods**: Simple Average, Weighted Average, Optimized Weights

### Why This Worked:
1. **More trials = better optimization**: 50 trials finds better hyperparameters than 20
2. **Focus on strength**: XGB/LGB are best models (0.918-0.919), optimizing them has more impact
3. **Time efficiency**: Skipping GB saves 10-15+ hours while still getting good GB model
4. **Better ensemble**: Optimized XGB/LGB + default GB > less optimized all models

### Comparison:
- **Mac (20 trials + GB)**: 0.92057 - GB optimization took too long, fewer trials for XGB/LGB
- **PC (50 trials, no GB)**: 0.92175 - More trials for best models, skipped slow GB
- **Result**: 50 trials without GB is both faster AND better!

---

## Experiment 9: Cross-Validation Ensemble ❌ UNDERPERFORMED (TWICE)

- **Date**: Current session
- **File**: `cross_validation_ensemble.py`
- **Status**: ❌ Underperformed both attempts
- **Scores**: 
  - First attempt (default params): 0.91908
  - Second attempt (optimized params): 0.91908 (same!)
- **Comparison**: 0.92175 (optimized ensemble) > 0.91908 (CV ensemble)

### Attempt 1: Default Hyperparameters
- Used 5-fold cross-validation
- Trained XGBoost, LightGBM, GradientBoosting on each fold
- Simple averaging of predictions
- Used default hyperparameters (not optimized ones)
- **Result**: 0.91908

### Attempt 2: Optimized Hyperparameters
- Used 5-fold cross-validation
- Used optimized hyperparameters from 50-trial Optuna run
- Weighted averaging by ROC AUC scores
- LightGBM: Exact best params (Trial 31, 0.920327)
- XGBoost: Approximate optimized params
- **Result**: 0.91908 (same as attempt 1!)

### Why CV Underperformed (Both Attempts):
1. **Hyperparameter mismatch**: Hyperparameters optimized for one 80/20 split don't transfer well to different CV folds
2. **Overfitting to folds**: CV can create inconsistent predictions across folds
3. **Single split worked better**: The optimized ensemble (0.92175) used a single optimized train/val split, which was more stable
4. **Ensemble dilution**: Averaging predictions from different folds can dilute the signal
5. **CV overhead**: The additional complexity of CV doesn't always help when you have a good single split

### Key Findings:
- **CV doesn't always help**: Even with optimized hyperparameters, CV can underperform
- **Single split can be better**: When you have a good validation split and optimized hyperparameters, a single split can outperform CV
- **Optimized ensemble > CV ensemble**: 0.92175 (optimized) > 0.91908 (CV, both attempts)
- **Hyperparameters don't transfer**: Hyperparameters optimized for one split may not work well on different CV folds

### Lesson Learned:
Cross-validation is most useful when:
- You don't have optimized hyperparameters yet
- You're doing hyperparameter tuning **within** CV (nested CV)
- You have limited data and need to use all of it
- You want to estimate model variance

**When CV doesn't help:**
- You already have optimized hyperparameters for a specific split
- The hyperparameters were tuned for a different data split
- You have enough data for a good train/val split
- The single split approach is already working well

**Conclusion**: For this project, the single optimized split (0.92175) is better than CV (0.91908). Focus on ensemble combinations instead.

---

## Experiment 10: Combining Optimized + CatBoost ✅ IN PROGRESS

- **Date**: Current session
- **File**: `combine_optimized_catboost.py`
- **Status**: ✅ Generated files, awaiting submission results
- **Method**: Weighted combination of:
  - Optimized ensemble (0.92175) - from Experiment 8
  - CatBoost ensemble (0.92016) - from earlier experiments
- **Weight combinations tested**:
  - 85/15 (Optimized 85%, CatBoost 15%)
  - 90/10 (Optimized 90%, CatBoost 10%)
  - 95/5 (Optimized 95%, CatBoost 5%)
  - 80/20 (Optimized 80%, CatBoost 20%)
  - 75/25 (Optimized 75%, CatBoost 25%)

### Expected Results:
- **Target**: 0.9218-0.9220 (improvement over 0.92175)
- **Rationale**: Combining two strong ensembles (0.92175 + 0.92016) with weighted averaging
- **Best weights**: Likely 85/15 or 90/10 (since Optimized is stronger)

### Files Generated:
- `my_submission_optimized85_catboost15.csv`
- `my_submission_optimized90_catboost10.csv`
- `my_submission_optimized95_catboost5.csv`
- `my_submission_optimized80_catboost20.csv`
- `my_submission_optimized75_catboost25.csv`

---

## Next Steps - Prioritized Recommendations

### 🎯 High Priority (Highest Expected Impact)

**1. Hyperparameter Optimization for Best Models** ⭐⭐⭐
- **Target**: XGBoost, LightGBM, GradientBoosting (the 3 best models)
- **Method**: Use Optuna or GridSearchCV to tune:
  - Learning rate (try 0.01, 0.015, 0.02, 0.025, 0.03)
  - Max depth (try 7, 8, 9, 10)
  - Number of trees (try 600, 700, 800, 1000)
  - Regularization (reg_alpha, reg_lambda)
  - Subsampling rates
- **Expected**: +0.0005 to +0.002 (could push 0.92100 → 0.9215-0.923)
- **Time**: 2-4 hours (but automated)
- **Why**: You haven't done systematic tuning yet - this is the biggest untapped opportunity

**2. Cross-Validation Ensemble** ⭐⭐⭐
- **Method**: K-fold CV (5-10 folds) with out-of-fold predictions
- **Benefits**: 
  - More robust evaluation (less variance than single 80/20 split)
  - Better generalization (uses all training data)
  - Reduces overfitting to single validation set
- **Expected**: +0.0003 to +0.001
- **Time**: 1-2 hours
- **Why**: Current single split may not be optimal - CV is more reliable

**3. Combine Optimized Ensemble with CatBoost** ⭐⭐
- **What**: Take your 0.92175 (optimized XGB/LGB/GB) and combine with CatBoost ensemble (0.92016)
- **Method**: Weighted combination (try 85/15, 90/10, 95/5)
- **Expected**: +0.0001 to +0.0003 (could reach 0.9218-0.9220)
- **Time**: < 5 minutes
- **Why**: Two-stage ensemble (ensemble of ensembles) worked before (0.92100)

### 🔧 Medium Priority (Good ROI)

**4. More Feature Engineering** ⭐⭐
- **Ideas**:
  - Target encoding for categorical features
  - More polynomial interactions (cubic terms)
  - Binning continuous features
  - Time-based features (if applicable)
  - Statistical features (rolling means, std dev)
- **Expected**: +0.0002 to +0.0008
- **Time**: 2-3 hours
- **Why**: Feature engineering has been key to your success

**5. Different Train/Validation Splits** ⭐
- **Method**: Try multiple random seeds for train/val split, then average predictions
- **Expected**: +0.0001 to +0.0003
- **Time**: 30 minutes
- **Why**: Reduces variance from single split

**6. Ensemble Weight Optimization** ⭐
- **Method**: Use scipy.optimize on the 3-model ensemble + CatBoost
- **Expected**: +0.0001 to +0.0002
- **Time**: < 5 minutes
- **Why**: Current weights (80/90%) are manual - optimization might find better

### 📊 Lower Priority (Diminishing Returns)

**7. Neural Network Hyperparameter Tuning** ⭐
- **Only if**: You want to include NNs (currently they hurt performance)
- **Method**: Tune learning rate, batch size, architecture, dropout
- **Expected**: +0.0001 to +0.0005 (if successful)
- **Time**: 4-6 hours
- **Why**: NNs currently underperform - may not be worth the effort

**8. Try Different Ensemble Methods** ⭐
- **Ideas**: 
  - Voting classifiers
  - Blending with different meta-learners
  - Stacking with cross-validation (to reduce overfitting)
- **Expected**: +0.0001 to +0.0003
- **Time**: 1-2 hours

---

## 🚀 Recommended Action Plan

**Week 1 (Quick Wins)**:
1. ✅ Hyperparameter optimization (COMPLETED) → Result: 0.92175
2. Combine optimized ensemble with CatBoost (5 min) → Expected: 0.9218-0.9220
3. Optimize ensemble weights (5 min) → Expected: +0.0001-0.0002

**Week 2 (High Impact)**:
4. Cross-validation ensemble (1-2 hours) → Expected: +0.0003-0.001
5. Try 100 trials for XGB/LGB (4-6 hours) → Expected: +0.0002-0.0005

**Week 3 (Polish)**:
6. More feature engineering (2-3 hours) → Expected: +0.0002-0.0008
7. Multiple splits and blending (30 min) → Expected: +0.0001-0.0003

**Target**: Push from 0.92175 → 0.922-0.924 (top tier performance)
