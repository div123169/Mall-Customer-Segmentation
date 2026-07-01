import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def clean_data(df):
    """
    Performs data cleaning: removes duplicates and imputes any missing values.
    """
    df_cleaned = df.copy()
    
    # Duplicate Analysis
    duplicates = df_cleaned.duplicated().sum()
    if duplicates > 0:
        print(f"Warning: Found {duplicates} duplicate rows. Removing duplicates...")
        df_cleaned = df_cleaned.drop_duplicates()
    
    # Missing Value Analysis & Imputation
    # For numerical columns, fill with median
    num_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        null_count = df_cleaned[col].isnull().sum()
        if null_count > 0:
            print(f"Imputing {null_count} missing values in numerical column: '{col}'")
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
            
    # For categorical columns, fill with mode
    cat_cols = df_cleaned.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        null_count = df_cleaned[col].isnull().sum()
        if null_count > 0:
            print(f"Imputing {null_count} missing values in categorical column: '{col}'")
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
            
    return df_cleaned

def encode_gender(df):
    """
    Encodes the 'Gender' column to numerical values:
    Female -> 0, Male -> 1
    """
    df_encoded = df.copy()
    if 'Gender' in df_encoded.columns:
        # Standardize strings to avoid casing issues
        df_encoded['Gender'] = df_encoded['Gender'].astype(str).str.strip().str.capitalize()
        df_encoded['Gender_Encoded'] = df_encoded['Gender'].map({'Female': 0, 'Male': 1})
        # Handle unexpected inputs gracefully by default filling to Female (0)
        df_encoded['Gender_Encoded'] = df_encoded['Gender_Encoded'].fillna(0).astype(int)
    else:
        raise ValueError("Missing required column 'Gender' in the dataset.")
    return df_encoded

def scale_features(df, features_to_scale, scaler=None):
    """
    Standardizes the specified numerical features using StandardScaler.
    If an existing scaler is provided, applies transform only (for inference).
    Otherwise, fits and transforms.
    
    Returns:
        df_scaled: Copy of dataframe with scaled features
        scaler: The StandardScaler instance used
    """
    df_scaled = df.copy()
    if scaler is None:
        scaler = StandardScaler()
        df_scaled[features_to_scale] = scaler.fit_transform(df_scaled[features_to_scale])
    else:
        df_scaled[features_to_scale] = scaler.transform(df_scaled[features_to_scale])
    return df_scaled, scaler

def preprocess_pipeline(df, features_to_scale=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'], scaler=None):
    """
    Runs the complete preprocessing pipeline on raw data:
    1. Clean duplicates and missing values.
    2. Encode 'Gender' to binary numeric values.
    3. Standardize selected features.
    
    Returns:
        df_preprocessed: Fully processed dataframe
        scaler: Scaler instance used for training
    """
    cleaned_df = clean_data(df)
    encoded_df = encode_gender(cleaned_df)
    scaled_df, scaler = scale_features(encoded_df, features_to_scale, scaler)
    return scaled_df, scaler
