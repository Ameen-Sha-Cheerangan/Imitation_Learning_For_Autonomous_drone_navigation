import os
import shutil
from PIL import Image

SRC_DIR = os.getcwd()  # Current directory
DST_DIR = os.path.join(SRC_DIR, "resized_data")

if not os.path.exists(DST_DIR):
    os.makedirs(DST_DIR)

for folder_name in os.listdir(SRC_DIR):
    src_folder = os.path.join(SRC_DIR, folder_name)
    dst_folder = os.path.join(DST_DIR, folder_name)
    if os.path.isdir(src_folder):
        # Copy folder structure and files except images
        if not os.path.exists(dst_folder):
            shutil.copytree(src_folder, dst_folder, ignore=shutil.ignore_patterns('images'))
        # Now handle images
        images_src = os.path.join(src_folder, 'images')
        images_dst = os.path.join(dst_folder, 'images')
        if os.path.exists(images_src):
            if not os.path.exists(images_dst):
                os.makedirs(images_dst)
            for img_file in os.listdir(images_src):
                src_img_path = os.path.join(images_src, img_file)
                dst_img_path = os.path.join(images_dst, img_file)
                # Only process if not already present
                if not os.path.exists(dst_img_path):
                    try:
                        with Image.open(src_img_path) as img:
                            img = img.convert('RGB')
                            img = img.resize((224, 224), Image.BICUBIC)
                            img.save(dst_img_path)
                    except Exception as e:
                        print(f"Error processing {src_img_path}: {e}")
