import os
import csv
import re

base_dir = os.getcwd()
folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

# Sort numerically based on leading digits in folder names (e.g., 1st, 2nd...)
folders = sorted(folders, key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else float('inf'))

print('Folders to check:', folders)

for folder in folders:
    images_path = os.path.join(base_dir, folder, "images")
    csv_path = os.path.join(base_dir, folder, "airsim_with_gaze_closest.csv")
    if os.path.exists(images_path) and os.path.exists(csv_path):
        image_files = [f for f in os.listdir(images_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        num_images = len(image_files)
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            num_rows = sum(1 for row in reader)
        num_csv_data = num_rows - 1  # subtract header row
        status = "OK" if num_images == num_csv_data else f"MISMATCH: {num_images} images, {num_csv_data} csv rows"
        print(f"{folder}: {status}")
    else:
        print(f"{folder}: images folder or CSV missing")
