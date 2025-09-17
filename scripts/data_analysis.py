import pandas as pd
import numpy as np

try:
    # Load the dataset from your local file
    df = pd.read_csv('data/raw_claims.csv')

    print("--- DataFrame Info ---")
    # Display basic information about the dataframe
    df.info()

    print("\n--- DataFrame Head (First 10 Rows) ---")
    # Display the first 10 rows
    print(df.head(10))

    print("\n--- Numerical Columns Description ---")
    # Display summary statistics for numerical columns
    print(df.describe())

    print("\n--- Value Counts for 'denial_reason' ---")
    # Display value counts for key categorical columns
    print(df['denial_reason'].value_counts(dropna=False))

    print("\n--- Value Counts for 'is_denied' ---")
    print(df['is_denied'].value_counts(dropna=False))

    print("\n--- Value Counts for 'payer_name' ---")
    print(df['payer_name'].value_counts(dropna=False))

    print("\n--- Value Counts for 'cpt_code' ---")
    print(df['cpt_code'].value_counts(dropna=False))

    print("\n--- Value Counts for 'provider_npi' ---")
    print(df['provider_npi'].value_counts(dropna=False))

except FileNotFoundError:
    print("Error: 'raw_claims.csv' not found. Please make sure the script is in the same directory as the CSV file.")
except Exception as e:
    print(f"An error occurred: {e}")