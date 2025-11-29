# Submission Verification Script
# Verifies that the submission file meets competition requirements

import pandas as pd
import sys

def verify_submission(submission_file="my_submission.csv"):
    """
    Verify that submission file meets competition requirements:
    - CSV format with header
    - Exactly 254569 rows (data rows, excluding header)
    - Columns: id, loan_paid_back
    - loan_paid_back values are probabilities (0.0 to 1.0)
    """
    
    print("="*80)
    print("SUBMISSION VERIFICATION")
    print("="*80)
    print(f"Checking file: {submission_file}\n")
    
    try:
        # Read the submission file
        df = pd.read_csv(submission_file)
        
        # Check 1: Number of rows
        num_rows = len(df)
        expected_rows = 254569
        print(f"✓ Number of data rows: {num_rows} (expected: {expected_rows})")
        if num_rows != expected_rows:
            print(f"  ✗ ERROR: Expected {expected_rows} rows, got {num_rows}")
            return False
        else:
            print(f"  ✓ Correct number of rows")
        
        # Check 2: Column names
        expected_columns = ['id', 'loan_paid_back']
        actual_columns = list(df.columns)
        print(f"\n✓ Columns: {actual_columns} (expected: {expected_columns})")
        if actual_columns != expected_columns:
            print(f"  ✗ ERROR: Column names don't match")
            return False
        else:
            print(f"  ✓ Column names are correct")
        
        # Check 3: ID column
        print(f"\n✓ ID column:")
        print(f"  - Type: {df['id'].dtype}")
        print(f"  - Unique values: {df['id'].nunique()} (expected: {num_rows})")
        if df['id'].nunique() != num_rows:
            print(f"  ✗ WARNING: Duplicate IDs found")
        else:
            print(f"  ✓ All IDs are unique")
        
        # Check 4: loan_paid_back column (probabilities)
        print(f"\n✓ loan_paid_back column:")
        print(f"  - Type: {df['loan_paid_back'].dtype}")
        print(f"  - Min value: {df['loan_paid_back'].min():.6f}")
        print(f"  - Max value: {df['loan_paid_back'].max():.6f}")
        print(f"  - Mean value: {df['loan_paid_back'].mean():.6f}")
        
        # Check if values are in valid range [0, 1]
        if df['loan_paid_back'].min() < 0.0 or df['loan_paid_back'].max() > 1.0:
            print(f"  ✗ ERROR: Values must be between 0.0 and 1.0")
            return False
        else:
            print(f"  ✓ All values are in valid range [0.0, 1.0]")
        
        # Check for NaN values
        nan_count = df['loan_paid_back'].isna().sum()
        if nan_count > 0:
            print(f"  ✗ ERROR: Found {nan_count} NaN values")
            return False
        else:
            print(f"  ✓ No NaN values")
        
        # Check 5: File format
        print(f"\n✓ File format:")
        print(f"  - Format: CSV")
        print(f"  - Has header: Yes")
        print(f"  - Total lines (including header): {num_rows + 1}")
        
        # Summary
        print("\n" + "="*80)
        print("✓ VERIFICATION PASSED")
        print("="*80)
        print(f"\nSubmission file '{submission_file}' is valid and ready for submission!")
        print(f"  - Rows: {num_rows} ✓")
        print(f"  - Columns: {expected_columns} ✓")
        print(f"  - Format: CSV with header ✓")
        print(f"  - Values: Probabilities in [0, 1] ✓")
        print("="*80 + "\n")
        
        return True
        
    except FileNotFoundError:
        print(f"✗ ERROR: File '{submission_file}' not found")
        return False
    except pd.errors.EmptyDataError:
        print(f"✗ ERROR: File '{submission_file}' is empty")
        return False
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    submission_file = sys.argv[1] if len(sys.argv) > 1 else "my_submission.csv"
    success = verify_submission(submission_file)
    sys.exit(0 if success else 1)

