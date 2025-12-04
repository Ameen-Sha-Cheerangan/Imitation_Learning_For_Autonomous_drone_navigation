import time
import eyeware.beam_eye_tracker as beam
import csv
import datetime  # 1. Import the datetime module

# --- Configuration ---
SCREEN_WIDTH = 640/2560
SCREEN_HEIGHT = 360/1440
APP_NAME = "QHD Gaze Reporter"
OUTPUT_FILE = "gaze_log.csv"  # Define the output filename

def main():
    """
    Initializes the Eyeware Beam API, polls for eye-tracking data,
    and logs both absolute and relative timestamps along with gaze coordinates
    to a CSV file.
    """
    eyeware_api = None
    try:
        p00 = beam.Point(0, 0)
        p11 = beam.Point(SCREEN_WIDTH, SCREEN_HEIGHT)
        viewport = beam.ViewportGeometry(point_00=p00, point_11=p11)

        print(f"Initializing Eyeware API for '{APP_NAME}'...")
        eyeware_api = beam.API(APP_NAME, viewport)
        print("API Initialized successfully.")

        eyeware_api.attempt_starting_the_beam_eye_tracker()
        
        # Open the CSV file to write to
        with open(OUTPUT_FILE, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            
            # 2. Update the CSV header to include both timestamp types
            header = ['absolute_timestamp_iso', 'relative_timestamp_s', 'x', 'y', 'confidence']
            csv_writer.writerow(header)
            print(f"Logging gaze data to '{OUTPUT_FILE}'...")

            last_timestamp = beam.NULL_DATA_TIMESTAMP()
            print("\nStarting to receive tracking data... Press Ctrl+C to stop.")

            while True:
                # Wait for new data to become available
                has_new_data = eyeware_api.wait_for_new_tracking_state_set(last_timestamp, timeout_ms=1000)

                if has_new_data:
                    # 3. Capture the absolute "wall-clock" time immediately upon receiving data
                    absolute_time_iso = datetime.datetime.now().isoformat()
                
                    # Get the full data payload
                    tracking_state_set = eyeware_api.get_latest_tracking_state_set()
                    user_state = tracking_state_set.user_state()
                    gaze_data = user_state.unified_screen_gaze

                    confidence = gaze_data.confidence
                    if confidence != beam.TrackingConfidence.LOST_TRACKING.value:
                        # Get the relative timestamp from the tracker
                        relative_timestamp = user_state.timestamp_in_seconds.value
                        gaze_point = gaze_data.point_of_regard
                        confidence_name = beam.TrackingConfidence(confidence).name

                        # 4. Create a data row with BOTH timestamps and write it to the file
                        data_row = [absolute_time_iso, relative_timestamp, gaze_point.x*SCREEN_WIDTH, gaze_point.y*SCREEN_HEIGHT, confidence_name]
                        csv_writer.writerow(data_row)
                        
                        # (Optional) Update the console printout for real-time feedback
                        print(f"Abs Time: {absolute_time_iso} | Gaze: ({gaze_point.x*SCREEN_WIDTH}, {gaze_point.y*SCREEN_HEIGHT}) | Conf: {confidence_name}")

                else:
                    # This is useful to see if the script is running but not receiving data
                    print("Waiting for new data...")
                
                # A small sleep prevents the loop from consuming too much CPU if there's an issue
                time.sleep(0.001)

    except (RuntimeError, ValueError) as e:
        print(f"ERROR: Failed to initialize the Eyeware API. {e}")
        print("Please ensure the Eyeware Beam application is running.")
    except KeyboardInterrupt:
        print(f"\nProgram interrupted. Data saved to '{OUTPUT_FILE}'.")
    finally:
        if eyeware_api:
            del eyeware_api
            print("Eyeware API has been shut down.")

if __name__ == "__main__":
    main()
