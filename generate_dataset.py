"""
Generate a messy ML experiments dataset for data cleaning RL task
"""
import pandas as pd
import numpy as np
import json
import random
from datetime import datetime, timedelta

def generate_messy_dataset(output_file="experiments.csv", seed=42):
    """
    Generate a dataset with intentional data quality issues
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Base clean data
    num_clean_rows = 60
    
    model_names_clean = ["BERT", "GPT-2", "ResNet", "LSTM", "Transformer", "CNN", "RNN"]
    statuses_valid = ["completed", "success", "finished"]
    statuses_invalid = ["failed", "running", "error", "pending"]
    
    clean_data = []
    
    for i in range(num_clean_rows):
        exp_id = f"exp_{i:04d}"
        model = random.choice(model_names_clean)
        accuracy = round(random.uniform(0.65, 0.95), 4)
        train_time = round(random.uniform(100, 5000), 2)
        date = (datetime.now() - timedelta(days=random.randint(1, 100))).strftime("%Y-%m-%d")
        hyperparams = json.dumps({
            "learning_rate": round(random.uniform(0.0001, 0.01), 5),
            "batch_size": random.choice([16, 32, 64, 128]),
            "epochs": random.randint(5, 50)
        })
        status = random.choice(statuses_valid)
        
        clean_data.append({
            "experiment_id": exp_id,
            "model_name": model,
            "accuracy": accuracy,
            "training_time": train_time,
            "date": date,
            "hyperparams": hyperparams,
            "status": status
        })
    
    df = pd.DataFrame(clean_data)
    
    # Now introduce issues
    issues_added = []
    
    # 1. Add duplicates (15 duplicate rows)
    duplicate_indices = random.sample(range(len(df)), 15)
    duplicates = df.iloc[duplicate_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    issues_added.append(f"Added 15 duplicate rows")
    
    # 2. Add missing values in accuracy (8 rows)
    missing_acc_idx = random.sample(range(len(df)), 8)
    df.loc[missing_acc_idx, 'accuracy'] = np.nan
    issues_added.append(f"Added 8 missing accuracy values")
    
    # 3. Add missing values in training_time (7 rows)
    missing_time_idx = random.sample(range(len(df)), 7)
    df.loc[missing_time_idx, 'training_time'] = np.nan
    issues_added.append(f"Added 7 missing training_time values")
    
    # 4. Add out-of-range accuracy values (5 rows > 1.0, 3 rows < 0)
    high_acc_idx = random.sample(range(len(df)), 5)
    df.loc[high_acc_idx, 'accuracy'] = [round(random.uniform(1.1, 2.5), 4) for _ in range(5)]
    low_acc_idx = random.sample(range(len(df)), 3)
    df.loc[low_acc_idx, 'accuracy'] = [round(random.uniform(-0.5, -0.01), 4) for _ in range(3)]
    issues_added.append(f"Added 8 out-of-range accuracy values")
    
    # 5. Add negative/zero training times (6 rows)
    neg_time_idx = random.sample(range(len(df)), 6)
    df.loc[neg_time_idx, 'training_time'] = [round(random.uniform(-1000, 0), 2) for _ in range(6)]
    issues_added.append(f"Added 6 non-positive training_time values")
    
    # 6. Add formatting issues to model_name (inconsistent case, whitespace)
    format_idx = random.sample(range(len(df)), 20)
    for idx in format_idx:
        model = df.loc[idx, 'model_name']
        # Random formatting issue
        formatting_type = random.choice(['lower', 'upper', 'spaces', 'mixed'])
        if formatting_type == 'lower':
            df.loc[idx, 'model_name'] = model.lower()
        elif formatting_type == 'upper':
            df.loc[idx, 'model_name'] = model.upper()
        elif formatting_type == 'spaces':
            df.loc[idx, 'model_name'] = f"  {model} "
        elif formatting_type == 'mixed':
            df.loc[idx, 'model_name'] = model.lower().capitalize()
    issues_added.append(f"Added formatting issues to 20 model_name values")
    
    # 7. Add invalid status values (10 rows)
    invalid_status_idx = random.sample(range(len(df)), 10)
    for idx in invalid_status_idx:
        df.loc[idx, 'status'] = random.choice(statuses_invalid)
    issues_added.append(f"Added 10 invalid status values")
    
    # Shuffle the dataframe to mix up the issues
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"Generated messy dataset: {output_file}")
    print(f"Total rows: {len(df)}")
    print("\nIssues introduced:")
    for issue in issues_added:
        print(f"  - {issue}")
    
    # Calculate expected clean size
    # This is approximate since issues can overlap
    print(f"\nExpected rows after cleaning: ~40-50 (varies due to overlapping issues)")
    
    return df

def analyze_dataset(filename="experiments.csv"):
    """
    Analyze the dataset and report issues
    """
    df = pd.read_csv(filename)
    
    print(f"\n{'='*60}")
    print(f"Dataset Analysis: {filename}")
    print(f"{'='*60}")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Columns: {list(df.columns)}")
    
    print(f"\nData Quality Issues:")
    
    # Duplicates
    dup_count = df.duplicated().sum()
    dup_by_id = df.duplicated(subset=['experiment_id'], keep=False).sum()
    print(f"  - Duplicate rows (exact): {dup_count}")
    print(f"  - Duplicate experiment_ids: {dup_by_id}")
    
    # Missing values
    print(f"  - Missing accuracy: {df['accuracy'].isna().sum()}")
    print(f"  - Missing training_time: {df['training_time'].isna().sum()}")
    
    # Out of range
    valid_acc = df[(df['accuracy'] >= 0) & (df['accuracy'] <= 1)]['accuracy']
    invalid_acc = len(df) - len(valid_acc) - df['accuracy'].isna().sum()
    print(f"  - Invalid accuracy (not in [0,1]): {invalid_acc}")
    
    valid_time = df[df['training_time'] > 0]['training_time']
    invalid_time = len(df) - len(valid_time) - df['training_time'].isna().sum()
    print(f"  - Invalid training_time (<=0): {invalid_time}")
    
    # Formatting
    properly_formatted = df[df['model_name'].str.strip() == df['model_name'].str.title().str.strip()]['model_name']
    format_issues = len(df) - len(properly_formatted)
    print(f"  - Model name formatting issues: {format_issues}")
    
    # Invalid status
    valid_statuses = ['completed', 'success', 'finished']
    invalid_status = df[~df['status'].isin(valid_statuses)]['status']
    print(f"  - Invalid status values: {len(invalid_status)}")
    
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # Generate the dataset
    df = generate_messy_dataset()
    
    # Analyze it
    analyze_dataset()
    
    print("\n✓ Dataset generated successfully!")
    print("  File: experiments.csv")
    print("  Ready for RL training task")
