# backend/harvest_data.py
# This script now reads a locally downloaded data file from the UPAg portal,
# filters it, and cleans it for use in the project.

import pandas as pd
import io
import os
import re

def clean_local_data():
    """
    Reads a raw, locally downloaded CSV file from UPAg, filters it for a specific
    crop, and saves a cleaned version.
    """
    # --- Step 1: Define the local input filename ---
    input_filename = 'raw_production_data.csv'

    print(f"Attempting to read local data file: '{input_filename}'")

    # Check if the user has downloaded the file first
    if not os.path.exists(input_filename):
        print("\n--- ERROR: Raw data file not found! ---")
        print(f"Please manually download the CSV from the UPAg portal, move it to the 'backend' folder,")
        print(f"and rename it to '{input_filename}' before running this script.")
        return None

    try:
        # --- Step 2: Load the Local Data into Pandas ---
        df = pd.read_csv(input_filename)
        print(f"Successfully loaded {len(df)} total records from the local file.")
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return None
    
    # --- Step 3: Clean and Filter the Data (Adapted for UPAg format) ---
    print("Cleaning and filtering data...")
    
    target_crop = 'Rice'
    
    # Dynamically find the year from the column names (e.g., 'Area-2023-24')
    year_str = None
    area_col = None
    prod_col = None
    for col in df.columns:
        if 'Area-' in col:
            area_col = col
            # Extract year like "2023" from "Area-2023-24"
            match = re.search(r'(\d{4})', col)
            if match:
                year_str = match.group(1)
        if 'Production-' in col:
            prod_col = col
            
    if not all([year_str, area_col, prod_col]):
        print(f"--- ERROR: Could not find year-specific Area and Production columns. ---")
        print(f"Found columns: {df.columns.tolist()}")
        return None

    print(f"Detected data for the year: {year_str}")
    target_year = int(year_str)

    # Ensure the required columns exist before trying to filter
    required_columns = ['Crop', area_col, prod_col]
    if not all(col in df.columns for col in required_columns):
        print(f"--- ERROR: The CSV file is missing required columns. ---")
        print(f"Expected columns like {required_columns}, but found: {df.columns.tolist()}")
        return None

    df_filtered = df[df['Crop'].str.strip().str.lower() == target_crop.lower()].copy()

    print(f"Found {len(df_filtered)} records for '{target_crop}'.")

    # Rename columns to a standard format
    df_filtered.rename(columns={
        'State': 'State',
        'District': 'District',
        'Crop': 'Crop',
        'Season': 'Season',
        area_col: 'Area_Hectare',
        prod_col: 'Production_Tonnes'
    }, inplace=True)
    
    # Add a 'Year' column
    df_filtered['Year'] = target_year

    df_filtered.dropna(subset=['Production_Tonnes'], inplace=True)
    df_filtered = df_filtered[df_filtered['Production_Tonnes'] > 0]

    df_filtered['Area_Hectare'] = pd.to_numeric(df_filtered['Area_Hectare'], errors='coerce')
    df_filtered['Production_Tonnes'] = pd.to_numeric(df_filtered['Production_Tonnes'], errors='coerce')

    # --- Step 4: Calculate Yield and Finalize the Dataset ---
    df_filtered['Yield_kg_per_hectare'] = (df_filtered['Production_Tonnes'] * 1000) / df_filtered['Area_Hectare']
    
    final_df = df_filtered[['Year', 'State', 'District', 'Yield_kg_per_hectare']].copy()
    final_df.dropna(inplace=True)

    print(f"Cleaned data has {len(final_df)} final records.")

    # --- Step 5: Save the Cleaned Data to a New CSV ---
    output_filename = 'cleaned_production_data.csv'
    final_df.to_csv(output_filename, index=False)
    
    print(f"\n--- SUCCESS ---")
    print(f"Cleaned data has been saved to '{output_filename}'")
    return final_df

# --- Run the Script ---
if __name__ == '__main__':
    cleaned_data = clean_local_data()
    if cleaned_data is not None:
        print("\n--- First 5 rows of the cleaned dataset: ---")
        print(cleaned_data.head())
