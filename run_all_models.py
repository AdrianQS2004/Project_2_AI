# Main Script to Run All Models and Compare Results
# Introduction to Artificial Intelligence
# Loan Default Prediction Competition
# By Team
# Copyright 2025, Texas Tech University - Costa Rica

import sys
import importlib.util
import os
from results_tracker import ResultsTracker

def run_model(script_path, model_name, model_type):
    """Run a model script and extract results"""
    print("\n" + "="*80)
    print(f"Running {model_name}")
    print("="*80)
    
    try:
        # Import and run the module
        spec = importlib.util.spec_from_file_location("model_module", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Try to extract results if available
        # Note: This requires models to set global variables or return results
        # For now, we'll just run them and track manually
        print(f"\n✓ {model_name} completed successfully")
        return True
    except Exception as e:
        print(f"\n✗ Error running {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run all models"""
    
    print("="*80)
    print("LOAN DEFAULT PREDICTION - MODEL COMPARISON")
    print("="*80)
    print("\nThis script will run all available models and track their results.")
    print("Note: You may need to manually record ROC AUC scores in results_tracker.py")
    print("="*80)
    
    # Initialize results tracker
    tracker = ResultsTracker("model_results.csv")
    
    # Define all models to run
    models = [
        {
            'script': 'BaselinePerformance1_method.py',
            'name': 'Baseline_1_LogisticRegression_PyTorch',
            'type': 'Baseline'
        },
        {
            'script': 'BaselinePerformance2_method.py',
            'name': 'Baseline_2_LogisticRegression_sklearn',
            'type': 'Baseline'
        },
        {
            'script': 'Classic_model_1_method.py',
            'name': 'Classic_1_RandomForest',
            'type': 'Classic'
        },
        {
            'script': 'Classic_model_2_method.py',
            'name': 'Classic_2_GradientBoosting',
            'type': 'Classic'
        },
        {
            'script': 'Classic_model_3_method.py',
            'name': 'Classic_3_XGBoost',
            'type': 'Classic'
        },
        {
            'script': 'Classic_model_4_method.py',
            'name': 'Classic_4_LightGBM',
            'type': 'Classic'
        },
        {
            'script': 'ImprovedNN_modelMethod.py',
            'name': 'NeuralNetwork_Improved',
            'type': 'Neural Network'
        },
    ]
    
    # Run each model
    results_summary = []
    
    for model in models:
        script_path = model['script']
        
        if not os.path.exists(script_path):
            print(f"\n⚠ Skipping {model['name']}: {script_path} not found")
            continue
        
        success = run_model(script_path, model['name'], model['type'])
        
        if success:
            results_summary.append({
                'name': model['name'],
                'type': model['type'],
                'status': 'Success'
            })
        else:
            results_summary.append({
                'name': model['name'],
                'type': model['type'],
                'status': 'Failed'
            })
    
    # Print summary
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    for result in results_summary:
        status_symbol = "✓" if result['status'] == 'Success' else "✗"
        print(f"{status_symbol} {result['name']:40s} [{result['type']:15s}] - {result['status']}")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Check the output above for ROC AUC scores from each model")
    print("2. Manually add results to the tracker using:")
    print("   from results_tracker import ResultsTracker")
    print("   tracker = ResultsTracker()")
    print("   tracker.add_result(...)")
    print("3. View summary with: tracker.print_summary()")
    print("4. Check 'my_submission.csv' for the latest predictions")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

