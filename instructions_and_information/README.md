# Instructions and Information

This folder contains all documentation, guides, and information files for the Loan Default Prediction Competition project.

## Documentation Files

### Getting Started
- **README.md** - Main project documentation and overview
- **QUICK_START.md** - Quick start guide for beginners
- **PROJECT_SUMMARY.md** - Summary of what's been implemented

### Experiment Tracking
- **EXPERIMENT_LOG.md** - Complete log of all experiments, scores, and changes
- **RESULTS_ANALYSIS.md** - Detailed analysis of results and findings

### Guides
- **IMPROVEMENT_GUIDE.md** - Guide to improve ROC AUC scores
- **NEXT_STEPS.md** - Next steps for improvement
- **PUSH_FOR_HIGHER.md** - Advanced techniques to push for higher scores
- **SUBMISSION_GUIDE.md** - Guide on which submission file to use

### Comparisons and Notes
- **COMPARISON.md** - Comparison of different ensemble methods
- **INSTALL_NOTES.md** - Installation notes and troubleshooting

## Quick Reference

### Current Best Score
**0.92100** - Weighted Ensemble (Advanced 80% + CatBoost 20%)

### Best Submission File
`submissions/my_submission_weighted_80.csv` or `submissions/my_submission_weighted_90.csv`

### Key Learnings
1. Feature engineering was key (Advanced ensemble)
2. Weighted ensembles > simple averaging
3. 80-90% weight on best model optimal
4. More features + more trees + lower learning rates = better

## File Organization

- All submission CSV files → `submissions/` folder
- All documentation → `instructions_and_information/` folder
- All Python scripts → Root folder

## Updates

This folder is automatically updated as experiments progress. Check `EXPERIMENT_LOG.md` for the latest results.
