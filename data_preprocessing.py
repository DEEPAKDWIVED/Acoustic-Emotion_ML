"""Data preprocessing module for emotion recognition."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from config import RANDOM_STATE, TEST_SIZE, VAL_TEST_SPLIT

# Global label encoder
label_encoder = LabelEncoder()


def load_data(csv_path):
    
    return pd.read_csv(csv_path)


def balance_data_with_smote(df, target_column="Emotion"):
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Apply SMOTE to balance the classes
    smote = SMOTE(random_state=RANDOM_STATE)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    
    # Create a new DataFrame with balanced data
    df_balanced = pd.DataFrame(X_balanced, columns=X.columns)
    df_balanced[target_column] = y_balanced
    
    print("Oversampling complete!")
    print("\nClass distribution after oversampling:")
    print(df_balanced[target_column].value_counts())
    
    return df_balanced


def split_data(X, y, test_size=TEST_SIZE, val_test_split=VAL_TEST_SPLIT, random_state=RANDOM_STATE):
    
    # First, split into train (70%) and temp (30% for validation + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Split temp into validation (15%) and test (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_test_split, random_state=random_state, stratify=y_temp
    )
    
    print("Data split complete!")
    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Validation set: {X_val.shape}, {y_val.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_data_pipeline(input_csv, output_csv=None, balance=True):
    
    # Load data
    print(f"Loading data from {input_csv}...")
    df = load_data(input_csv)
    
    # Balance data if requested
    if balance:
        print("\nBalancing data with SMOTE...")
        df = balance_data_with_smote(df)
        
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"Balanced data saved to {output_csv}")
    
    # Separate features and target
    X = df.drop(columns=["Emotion"])
    y = df["Emotion"]
    
    # Encode labels to numeric values
    print("\nEncoding labels...")
    global label_encoder
    y_encoded = label_encoder.fit_transform(y)
    print(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    # Split data
    print("\nSplitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y_encoded)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def decode_labels(y_encoded):
    """Convert encoded labels back to original string labels.
    
    Args:
        y_encoded: Encoded numeric labels
    
    Returns:
        Original string labels
    """
    return label_encoder.inverse_transform(y_encoded)


def get_label_mapping():
    """Get the mapping of labels to encoded values.
    
    Returns:
        Dictionary mapping original labels to encoded values
    """
    if hasattr(label_encoder, 'classes_'):
        return dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    return None


if __name__ == "__main__":
    from config import OUTPUT_CSV_ORIGINAL, OUTPUT_CSV_BALANCED
    
    # Run preprocessing pipeline
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_pipeline(
        OUTPUT_CSV_ORIGINAL, 
        OUTPUT_CSV_BALANCED, 
        balance=True
    )
