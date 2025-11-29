# Results Tracker for Loan Default Prediction Competition
# Tracks all model experiments, hyperparameters, and ROC AUC scores

import pandas as pd
import json
from datetime import datetime
import os

class ResultsTracker:
    def __init__(self, results_file="model_results.csv"):
        self.results_file = results_file
        self.results = []
        
        # Load existing results if file exists
        if os.path.exists(results_file):
            try:
                self.results_df = pd.read_csv(results_file)
                self.results = self.results_df.to_dict('records')
            except:
                self.results_df = pd.DataFrame()
        else:
            self.results_df = pd.DataFrame()
    
    def add_result(self, model_name, model_type, roc_auc_train=None, roc_auc_val=None, 
                   roc_auc_test=None, hyperparameters=None, notes="", submission_file=""):
        """
        Add a result entry
        
        Parameters:
        - model_name: Name of the model (e.g., "LogisticRegression_v1")
        - model_type: Type of model (e.g., "Baseline", "Classic", "Neural Network")
        - roc_auc_train: ROC AUC on training set
        - roc_auc_val: ROC AUC on validation set
        - roc_auc_test: ROC AUC on test set (if available)
        - hyperparameters: Dictionary of hyperparameters used
        - notes: Additional notes about the experiment
        - submission_file: Path to submission file generated
        """
        result = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_name': model_name,
            'model_type': model_type,
            'roc_auc_train': roc_auc_train,
            'roc_auc_val': roc_auc_val,
            'roc_auc_test': roc_auc_test,
            'hyperparameters': json.dumps(hyperparameters) if hyperparameters else "",
            'notes': notes,
            'submission_file': submission_file
        }
        
        self.results.append(result)
        self._save_results()
    
    def _save_results(self):
        """Save results to CSV file"""
        self.results_df = pd.DataFrame(self.results)
        self.results_df.to_csv(self.results_file, index=False)
    
    def get_best_model(self, metric='roc_auc_val'):
        """Get the best model based on validation ROC AUC"""
        if len(self.results) == 0:
            return None
        
        df = pd.DataFrame(self.results)
        df = df[df[metric].notna()]
        if len(df) == 0:
            return None
        
        best_idx = df[metric].idxmax()
        return df.loc[best_idx].to_dict()
    
    def print_summary(self):
        """Print a summary of all results"""
        if len(self.results) == 0:
            print("No results recorded yet.")
            return
        
        df = pd.DataFrame(self.results)
        print("\n" + "="*80)
        print("MODEL RESULTS SUMMARY")
        print("="*80)
        print(f"\nTotal experiments: {len(df)}")
        
        # Group by model type
        print("\nResults by Model Type:")
        print("-" * 80)
        for model_type in df['model_type'].unique():
            type_df = df[df['model_type'] == model_type]
            print(f"\n{model_type}:")
            for _, row in type_df.iterrows():
                val_auc = row['roc_auc_val'] if pd.notna(row['roc_auc_val']) else "N/A"
                train_auc = row['roc_auc_train'] if pd.notna(row['roc_auc_train']) else "N/A"
                print(f"  {row['model_name']:30s} | Train AUC: {train_auc:6.4f} | Val AUC: {val_auc:6.4f}")
        
        # Best model
        best = self.get_best_model()
        if best:
            print("\n" + "="*80)
            print("BEST MODEL (by Validation ROC AUC):")
            print("-" * 80)
            print(f"Model Name: {best['model_name']}")
            print(f"Model Type: {best['model_type']}")
            print(f"Train ROC AUC: {best['roc_auc_train']:.4f}")
            print(f"Val ROC AUC: {best['roc_auc_val']:.4f}")
            if best['hyperparameters']:
                print(f"Hyperparameters: {best['hyperparameters']}")
            print(f"Submission File: {best['submission_file']}")
            print("="*80 + "\n")

