import pandas as pd
from datetime import datetime

def process_flight_data(file_path, output_path=None):
    # Read the TXT file with tab separator
    df = pd.read_csv(file_path, sep='\t')
    
    # Convert Unix timestamp (milliseconds) to UTC datetime
    df['absolute_timestamp_iso'] = pd.to_datetime(df['TimeStamp'], unit='ms', utc=True)
    
    # Convert from UTC to IST (Indian Standard Time)
    df['absolute_timestamp_iso'] = df['absolute_timestamp_iso'].dt.tz_convert('Asia/Kolkata')
    
    # Remove timezone info to make it naive
    df['absolute_timestamp_iso'] = df['absolute_timestamp_iso'].dt.tz_localize(None)
    
    # Convert to ISO format with 'T' separator 
    df['absolute_timestamp_iso'] = df['absolute_timestamp_iso'].dt.strftime('%Y-%m-%dT%H:%M:%S.%f')
    
    # Display first few rows
    print("First 5 rows with new timestamp column:")
    print(df[['VehicleName', 'TimeStamp', 'absolute_timestamp_iso']].head())
    
    # Save to file if output path provided
    if output_path:
        df.to_csv(output_path, sep='\t', index=False)
        print(f"\nData saved to: {output_path}")
    
    return df


# Example usage with corrected Windows file path:
if __name__ == "__main__":
    # Using raw string (r"") to handle Windows backslashes properly
    input_file = r'C:\Users\ameen\Desktop\Drone Project\Gaze Tracking\airsim_rec.txt'
    
    output_file = 'flight_data_with_iso_timestamp.csv'
    
    # Process the data
    try:
        df = process_flight_data(input_file, output_file)
        print("\nProcessing completed successfully!")
        
        # Show data types and shape
        print("\nData types:")
        print(df.dtypes)
        print(f"\nDataFrame shape: {df.shape}")
        
        # Show sample of new column
        print("\nSample of absolute_timestamp_iso column:")
        print(df['absolute_timestamp_iso'].head().tolist())
        
    except FileNotFoundError:
        print(f"File {input_file} not found. Please check the file path.")
    except Exception as e:
        print(f"An error occurred: {e}")
