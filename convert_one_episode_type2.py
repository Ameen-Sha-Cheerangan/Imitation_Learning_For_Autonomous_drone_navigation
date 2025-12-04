# prepare_aril_like_from_airsim.py
"""
Create GRIL-compatible NPZ for gaze-regularized IL:
[images, depth, action, gaze_coord]
from an AirSim-style folder with:
  images/  (RGB PNG/JPEG)
  depth/   (depth PNG)
  airsim_with_gaze_closest.csv
"""

import csv
import os

import cv2
import numpy as np


def reshape_image(image):
    """Resize RGB to 224x224 and scale to [0,1], 3 channels."""
    width, height = 224, 224
    frame = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return frame.astype(np.float32) / 255.0


def reshape_depth(depth_img):
    """Resize depth to 224x224, grayscale + channel dim, scaled to [0,1]."""
    width, height = 224, 224
    frame = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    frame = np.expand_dims(frame, axis=2)
    return frame.astype(np.float32) / 255.0


def quaternion_to_euler(q_w, q_x, q_y, q_z):
    """Convert AirSim quaternion to roll, pitch, yaw (radians)."""
    ysqr = q_y * q_y

    t0 = +2.0 * (q_w * q_x + q_y * q_z)
    t1 = +1.0 - 2.0 * (q_x * q_x + ysqr)
    roll = np.arctan2(t0, t1)

    t2 = +2.0 * (q_w * q_y - q_z * q_x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)

    t3 = +2.0 * (q_w * q_z + q_x * q_y)
    t4 = +1.0 - 2.0 * (ysqr + q_z * q_z)
    yaw = np.arctan2(t3, t4)

    return roll, pitch, yaw


def prepare_data(data_path):
    """
    data_path: e.g. .../Data to be processed/1
    Inside it, expect:
      images/
      depth/
      airsim_with_gaze_closest.csv
    """
    dirname = data_path
    print(f"Processing episode folder: {dirname}")

    images_dir = os.path.join(dirname, "images")
    depth_dir = os.path.join(dirname, "depth")
    csv_path = os.path.join(dirname, "airsim_with_gaze_closest.csv")

    if not (os.path.isdir(images_dir) and os.path.isdir(depth_dir) and os.path.exists(csv_path)):
        print(f"Missing images/depth/csv in {dirname}")
        return

    imgs = []
    depth = []
    gaze_pos = []
    act_lbls = []
    prev_pos_z = None

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_name = row["ImageFile"]
            names = raw_name.split(";")
            rgb_name = names[0]
            depth_name = names[1]          # second camera image

            img_path = os.path.join(images_dir, rgb_name)
            depth_path = os.path.join(depth_dir, depth_name)

            if not os.path.exists(img_path) or not os.path.exists(depth_path):
                print(f"Missing image/depth, skipping: {img_path}, {depth_path}")
                continue

            # RGB image
            im = cv2.imread(img_path, cv2.IMREAD_COLOR)
            im = reshape_image(im)
            imgs.append(im)

            # depth image
            dt = cv2.imread(depth_path, cv2.IMREAD_COLOR)
            dt = reshape_depth(dt)
            depth.append(dt)

            # normalized gaze coordinates (AirSim 640x360)
            gx = float(row["x"]) / 640.0
            gy = float(row["y"]) / 360.0
            gaze_pos.append([gx, gy])

            # roll, pitch, yaw from quaternion
            q_w = float(row["Q_W"])
            q_x = float(row["Q_X"])
            q_y = float(row["Q_Y"])
            q_z = float(row["Q_Z"])
            roll, pitch, yaw = quaternion_to_euler(q_w, q_x, q_y, q_z)

            # throttle as change in altitude
            pos_z = float(row["POS_Z"])
            throttle = pos_z - prev_pos_z if prev_pos_z is not None else 0.0
            prev_pos_z = pos_z

            act_lbls.append([roll, pitch, throttle, yaw])

    if len(imgs) == 0:
        print("No valid samples, skipping NPZ.")
        return

    imgs = np.asarray(imgs, dtype=np.float32)         # [N,224,224,3]
    depth = np.asarray(depth, dtype=np.float32)       # [N,224,224,1]
    gaze_pos = np.asarray(gaze_pos, dtype=np.float32) # [N,2]
    act_lbls = np.asarray(act_lbls, dtype=np.float32) # [N,4]

    print("Final shapes before reshape:")
    print("  images:", imgs.shape)
    print("  depth:", depth.shape)
    print("  gaze_pos:", gaze_pos.shape)
    print("  act_lbls:", act_lbls.shape)

    # match GRIL ARIL format: [N,2,1] and [N,4,1]
    gaze_pos = gaze_pos.reshape(gaze_pos.shape[0], 2, 1)
    act_lbls = act_lbls.reshape(act_lbls.shape[0], 4, 1)

    npz_name = os.path.basename(os.path.normpath(data_path))  # e.g. "1"
    out_name = f"{npz_name}.npz"
    print(f"Saving {out_name}")

    np.savez_compressed(
        out_name,
        images=imgs,
        depth=depth,
        action=act_lbls,
        gaze_coords=gaze_pos,
    )
    print(f"Saved {out_name}")


if __name__ == "__main__":
    # For testing: convert the single episode folder "1"
    data_path = "101"   # or r"C:\full\path\to\1"
    prepare_data(data_path)
