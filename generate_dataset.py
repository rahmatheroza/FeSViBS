import pandas as pd
from sklearn.model_selection import train_test_split
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="Path to the input CSV file")
    parser.add_argument("--output_csv", required=True, help="Path to save the output CSV file")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of test data (default: 0.2)")
    
    args = parser.parse_args()
    generate_dataset(args.input_csv, args.output_csv, args.test_size)
