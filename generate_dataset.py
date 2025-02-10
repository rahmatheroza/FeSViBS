import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import argparse

def generate_dataset(input_csv, output_csv, test_size=0.2):
    # Load CSV file
    df = pd.read_csv(input_csv)
    
    # Ensure required columns exist
    required_columns = {"image", "target", "client"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Input CSV must contain columns: {required_columns}")
    
    # Stratified split into train and test
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["target"], random_state=42
    )
    
    # Assign fold labels
    train_df["fold"] = "train"
    test_df["fold"] = "test"
    
    # Concatenate back
    df = pd.concat([train_df, test_df]).reset_index(drop=True)
    
    # Create fold2 column
    df["fold2"] = df["fold"] + "_" + df["client"].astype(str)
    
    # Save the new dataset
    df.to_csv(output_csv, index=False)
    print(f"Dataset saved to {output_csv}")

def generate_k_fold_datasets(input_csv, output_prefix, test_size=0.2):
    # Load CSV file
    df = pd.read_csv(input_csv)
    
    # Ensure required columns exist
    required_columns = {"image", "target", "client"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Input CSV must contain columns: {required_columns}")
    
    # Determine number of folds (k = 1/test_size)
    k = int(1 / test_size)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(df, df["target"])):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        
        # Assign fold labels
        train_df["fold"] = "train"
        test_df["fold"] = "test"
        
        # Concatenate back
        fold_df = pd.concat([train_df, test_df]).reset_index(drop=True)
        
        # Create fold2 column
        fold_df["fold2"] = fold_df["fold"] + "_" + fold_df["client"].astype(str)
        
        # Save each fold dataset
        output_csv = f"{output_prefix}_fold{fold + 1}.csv"
        fold_df.to_csv(output_csv, index=False)
        print(f"Fold {fold + 1} dataset saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="Path to the input CSV file")
    parser.add_argument("--output_csv", required=True, help="Path to save the output CSV file")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of test data (default: 0.2)")
    parser.add_argument("--cross_validation", type=bool, default=False, help="If generate dataset into k=1/test files (default: False)")
    
    args = parser.parse_args()
    if args.cross_validation:
        generate_k_fold_datasets(args.input_csv, args.output_csv, args.test_size)
    else:
        generate_dataset(args.input_csv, args.output_csv, args.test_size)
