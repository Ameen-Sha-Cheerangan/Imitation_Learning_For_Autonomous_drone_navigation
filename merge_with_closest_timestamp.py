import pandas as pd

# Define the specific timestamp formats for each file
# Updated AirSim format: no timezone part and with 'T' separator
AIRSIM_TS_FORMAT = '%Y-%m-%dT%H:%M:%S.%f'
GAZE_TS_FORMAT = '%Y-%m-%dT%H:%M:%S.%f'

# Load files
try:
    # Use tab separator for AirSim file as before
    airsim = pd.read_csv('flight_data_with_iso_timestamp.csv', sep='\t')
    
    gaze = pd.read_csv('gaze_log.csv')
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    exit()

# Convert timestamps using the specific formats

# AirSim timestamps are now naive timestamps without timezone info, so no utc=True
airsim['absolute_timestamp_iso'] = pd.to_datetime(
    airsim['absolute_timestamp_iso'],
    format=AIRSIM_TS_FORMAT,
    errors='coerce'
)

# Gaze timestamps parsed with utc=True (timezone-aware)
gaze['absolute_timestamp_iso'] = pd.to_datetime(
    gaze['absolute_timestamp_iso'],
    format=GAZE_TS_FORMAT,
    errors='coerce',
    utc=True
)

# To merge correctly, convert gaze timestamps to naive (remove timezone info)
gaze['absolute_timestamp_iso'] = gaze['absolute_timestamp_iso'].dt.tz_localize(None)

# Drop any rows where timestamp conversion failed
airsim.dropna(subset=['absolute_timestamp_iso'], inplace=True)
gaze.dropna(subset=['absolute_timestamp_iso'], inplace=True)

# Check if dataframes are empty before proceeding
if airsim.empty or gaze.empty:
    print("\nERROR: One or both dataframes are still empty after parsing. Cannot merge.")
    print("Please check the AirSim file to confirm it is tab-separated.")
    exit()

# Sort values (required for merge_asof)
airsim = airsim.sort_values('absolute_timestamp_iso')
gaze = gaze.sort_values('absolute_timestamp_iso')

# --- Perform the merge ---
# Find the absolute nearest gaze point for each airsim point
merged = pd.merge_asof(
    airsim,
    gaze[['absolute_timestamp_iso', 'x', 'y', 'confidence']],
    on='absolute_timestamp_iso',
    direction='nearest'  # Using 'nearest' with no tolerance for the first attempt
)

# Save the final merged result
merged.to_csv('airsim_with_gaze_closest.csv', index=False)

print("Script finished successfully. Merged data saved to 'airsim_with_gaze_closest.csv'.")
print(f"Total rows in merged file: {len(merged)}")
print("Merged file head:")
print(merged.head())