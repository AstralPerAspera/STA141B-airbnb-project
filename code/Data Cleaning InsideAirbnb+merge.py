import gzip
import pandas as pd
import os

def clean_airbnb_data(input_file, city_name):
    """
    Clean Airbnb data for a single city
    
    Args:
        input_file (str): Input file path
        city_name (str): City name
    
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    print(f"\nProcessing city: {city_name}")
    print("=" * 60)
    
    try:
        # Read gz file
        print(f"Reading file: {input_file}")
        with gzip.open(input_file, 'rt', encoding='utf-8', errors='ignore') as f:
            # Read CSV file, only keep required columns
            df = pd.read_csv(f, usecols=[
                'price', 'latitude', 'longitude', 'room_type', 
                'accommodates', 'review_scores_rating', 'availability_365'
            ])
        
        print(f"Original data rows: {len(df)}")
        print(f"Data columns: {list(df.columns)}")
        
        # Data cleaning
        # 1. Convert price to numeric
        print("Processing price column...")
        if 'price' in df.columns:
            print(f"Price column type: {df['price'].dtype}")
            print(f"First 5 price values: {df['price'].head().tolist()}")
            
            # Process price column
            df['price'] = df['price'].astype(str)
            df['price'] = df['price'].str.replace('$', '', regex=False)
            df['price'] = df['price'].str.replace(',', '', regex=False)
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            print(f"First 5 processed price values: {df['price'].head().tolist()}")
        else:
            print("Warning: price column does not exist")
        
        # 2. Remove rows with missing latitude or longitude
        print("Removing missing values...")
        before_drop = len(df)
        df = df.dropna(subset=['latitude', 'longitude'])
        print(f"Removed {before_drop - len(df)} rows with missing coordinates")
        
        # 3. Handle price column (keep data if all are NaN)
        if 'price' in df.columns:
            price_nan_count = df['price'].isna().sum()
            if price_nan_count == len(df):
                print("Warning: All price values are NaN, will keep data with empty price column")
            else:
                # Remove rows with missing price
                before_drop = len(df)
                df = df.dropna(subset=['price'])
                print(f"Removed {before_drop - len(df)} rows with missing price")
        
        # 4. Remove rows with availability_365 = 0
        print("Removing rows with availability_365=0...")
        before_drop = len(df)
        df = df[df['availability_365'] != 0]
        print(f"Removed {before_drop - len(df)} rows with availability_365=0")
        
        # 5. Ensure latitude and longitude are numeric
        print("Processing latitude and longitude columns...")
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        
        # 6. Standardize column names
        print("Standardizing column names...")
        df = df.rename(columns={'review_scores_rating': 'review_score'})
        
        # 7. Add indicator variable is_entire_home
        print("Adding is_entire_home column...")
        df['is_entire_home'] = df['room_type'].apply(lambda x: 1 if x == 'Entire home/apt' else 0)
        
        # 8. Add city column
        print("Adding city column...")
        df['city'] = city_name
        
        print(f"Cleaned data rows: {len(df)}")
        print("=" * 60)
        
        return df
        
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """
    Main function
    """
    print("Starting Airbnb data cleaning")
    print("=" * 60)
    
    # Define file and city mapping
    files_cities = {
        'Los Angeles.gz': 'Los Angeles',
        'San Deigo.gz': 'San Diego',
        'San Francisco.gz': 'San Francisco'
    }
    
    # Process each city's data
    dfs = []
    for file_path, city_name in files_cities.items():
        if os.path.exists(file_path):
            df = clean_airbnb_data(file_path, city_name)
            if df is not None:
                dfs.append(df)
        else:
            print(f"File does not exist: {file_path}")
    
    # Merge data
    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        print(f"\nTotal merged data rows: {len(merged_df)}")
        
        # Save cleaned data
        output_file = 'cleaned_airbnb_listings_updated.csv'
        merged_df.to_csv(output_file, index=False)
        print(f"\nCleaned data saved to: {output_file}")
        
        # Display data information
        print("\nData information:")
        print(merged_df.info())
        
        print("\nData preview:")
        print(merged_df.head())
        
        return merged_df
    else:
        print("No files processed successfully")
        return None

if __name__ == "__main__":
    main()