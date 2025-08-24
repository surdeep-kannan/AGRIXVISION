# backend/enrich_data.py
# This script reads the cleaned production data and enriches it with
# historical satellite data (NDVI & Rainfall) from Google Earth Engine.

import pandas as pd
import ee
import time
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import os

def enrich_with_gee_data():
    """
    Reads cleaned production data, geocodes districts to get coordinates,
    and fetches corresponding historical satellite data from GEE.
    """
    # --- Step 1: Initialize GEE and Geocoder ---
    try:
        # Authenticate to Google Earth Engine
        # Make sure you have run 'earthengine authenticate' in your terminal
        ee.Initialize()
        print("Google Earth Engine authentication successful.")
    except Exception as e:
        print(f"Error initializing Earth Engine: {e}")
        return

    # Initialize the geocoder to find coordinates for district names
    geolocator = Nominatim(user_agent="agri_yield_app_v1")
    # Use RateLimiter to avoid overwhelming the geocoding service (1 request per second)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, error_wait_seconds=10)
    print("Geocoder initialized.")

    # --- Step 2: Load the Cleaned Production Data ---
    input_filename = 'cleaned_production_data.csv'
    try:
        df = pd.read_csv(input_filename)
        print(f"Successfully loaded '{input_filename}' with {len(df)} records.")
    except FileNotFoundError:
        print(f"--- ERROR: Input file not found ---")
        print(f"Please run 'harvest_data.py' first to create '{input_filename}'.")
        return

    # --- Step 3: Loop Through Data and Fetch Satellite Features ---
    satellite_data = []
    print("\nStarting data enrichment process. This may take several minutes...")

    for index, row in df.iterrows():
        try:
            # Construct a query string for the geocoder
            location_query = f"{row['District']}, {row['State']}, India"
            
            # Get coordinates for the district
            location = geocode(location_query)
            
            if location is None:
                print(f"[{index+1}/{len(df)}] - Could not find coordinates for: {location_query}. Skipping.")
                satellite_data.append({'mean_ndvi': None, 'total_rainfall_mm': None})
                continue

            lat, lon = location.latitude, location.longitude
            
            # Define a 10km square Area of Interest (AOI) around the district's center
            buffer = 0.05
            aoi = ee.Geometry.Rectangle([lon - buffer, lat - buffer, lon + buffer, lat + buffer])
            
            # Define the date range based on the year from the data
            year = int(row['Year'])
            start_date = f'{year}-06-01' # Rice growing season (Kharif)
            end_date = f'{year}-10-31'

            # Fetch NDVI data from Sentinel-2
            sentinel_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                                   .filterBounds(aoi).filterDate(start_date, end_date)
                                   .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 25)))
            get_ndvi = lambda image: image.normalizedDifference(['B8', 'B4']).rename('NDVI')
            mean_ndvi = sentinel_collection.map(get_ndvi).mean().reduceRegion(
                reducer=ee.Reducer.mean(), geometry=aoi, scale=100).get('NDVI')

            # Fetch Rainfall data from CHIRPS
            rainfall_collection = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
                                   .filterBounds(aoi).filterDate(start_date, end_date))
            total_rainfall = rainfall_collection.sum().reduceRegion(
                reducer=ee.Reducer.sum(), geometry=aoi, scale=5566).get('precipitation')

            # Execute the GEE computation
            ndvi_value = mean_ndvi.getInfo()
            rainfall_value = total_rainfall.getInfo()
            
            print(f"[{index+1}/{len(df)}] - Fetched data for {location_query}: NDVI={ndvi_value}, Rainfall={rainfall_value}")
            satellite_data.append({'mean_ndvi': ndvi_value, 'total_rainfall_mm': rainfall_value})

        except Exception as e:
            print(f"[{index+1}/{len(df)}] - An error occurred for {location_query}: {e}. Skipping.")
            satellite_data.append({'mean_ndvi': None, 'total_rainfall_mm': None})

    # --- Step 4: Merge Satellite Data and Save Final Dataset ---
    # Create a new DataFrame from the fetched satellite data
    satellite_df = pd.DataFrame(satellite_data)

    # Combine the original data with the new satellite data
    final_df = pd.concat([df, satellite_df], axis=1)
    
    # Drop rows where we couldn't fetch satellite data
    final_df.dropna(inplace=True)

    print(f"\nSuccessfully enriched {len(final_df)} records.")

    # Save the complete training dataset to a new file
    output_filename = 'training_dataset.csv'
    final_df.to_csv(output_filename, index=False)
    
    print(f"\n--- SUCCESS ---")
    print(f"Final training dataset has been saved to '{output_filename}'")
    return final_df

# --- Run the Script ---
if __name__ == '__main__':
    enriched_data = enrich_with_gee_data()
    if enriched_data is not None:
        print("\n--- First 5 rows of the final training dataset: ---")
        print(enriched_data.head())
