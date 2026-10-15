import pandas as pd
import numpy as np
import json
import os

print("--- Starting EconoCausal Data Preprocessing ---")

# 1. Load Raw JSON Datasets
portfolio = pd.read_json('Data/Raw/portfolio.json', orient='records', lines=True)
profile = pd.read_json('Data/Raw/profile.json', orient='records', lines=True)
transcript = pd.read_json('Data/Raw/transcript.json', orient='records', lines=True)

print("Raw datasets loaded successfully!")

# Helper function to convert unhashable types (lists, dicts) to strings for deduplication
def stringify_complex_cols(df):
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
            df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)
    return df

# 2. Clean Profile
profile.drop_duplicates(inplace=True)
profile['age'] = profile['age'].replace(118, np.nan)
profile['became_member_on'] = pd.to_datetime(profile['became_member_on'], format='%Y%m%d', errors='coerce')

# 3. Clean Portfolio
portfolio = stringify_complex_cols(portfolio)
portfolio.drop_duplicates(inplace=True)

# 4. Clean Transcript
transcript = stringify_complex_cols(transcript)
transcript.drop_duplicates(inplace=True)

# 5. Create Processed Directory
os.makedirs('Data/Processed', exist_ok=True)

# 6. Save Cleaned Datasets to CSV
profile.to_csv('Data/Processed/cleaned_profile.csv', index=False)
portfolio.to_csv('Data/Processed/cleaned_portfolio.csv', index=False)
transcript.to_csv('Data/Processed/cleaned_transcript.csv', index=False)

print("--- Preprocessing Complete: Cleaned files saved to Data/Processed/ ---")