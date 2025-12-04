
import os
from pathlib import Path

def count_files_in_folder(folder_path):
    """Count files (not folders) in a directory"""
    if not os.path.exists(folder_path):
        return 0
    try:
        return sum(1 for item in os.listdir(folder_path) 
                   if os.path.isfile(os.path.join(folder_path, item)))
    except PermissionError:
        return 0

def main():
    # Get the current directory (where script is placed)
    base_dir = os.getcwd()

    total_images = 0
    total_depth = 0
    folders_found = 0

    print("=" * 70)
    print("FOLDER ANALYSIS REPORT")
    print("=" * 70)
    print(f"{'Folder':<10} {'Images':<15} {'Depth':<15} {'Total':<10}")
    print("-" * 70)

    # Check folders 1 to 100
    for i in range(1, 101):
        folder_name = str(i)
        folder_path = os.path.join(base_dir, folder_name)

        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            # Check images folder
            images_path = os.path.join(folder_path, "images")
            images_count = count_files_in_folder(images_path)

            # Check depth folder
            depth_path = os.path.join(folder_path, "depth")
            depth_count = count_files_in_folder(depth_path)

            folder_total = images_count + depth_count

            # Only print if folder exists
            if images_count > 0 or depth_count > 0:
                print(f"{folder_name:<10} {images_count:<15} {depth_count:<15} {folder_total:<10}")
                total_images += images_count
                total_depth += depth_count
                folders_found += 1

    print("=" * 70)
    print(f"{'TOTALS':<10} {total_images:<15} {total_depth:<15} {total_images + total_depth:<10}")
    print("=" * 70)
    print(f"\nFolders processed: {folders_found}")
    print(f"Total files in 'images' folders: {total_images}")
    print(f"Total files in 'depth' folders: {total_depth}")
    print(f"Grand total: {total_images + total_depth}")

if __name__ == "__main__":
    main()
