import os
import numpy as np
import csv
from PIL import Image

def quaternion_to_euler(q_w, q_x, q_y, q_z):
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

# --- CONFIGURATION ---
episode = '1'
image_size = 224
base_dir = os.getcwd()
episode_dir = os.path.join(base_dir, episode)
images_dir = os.path.join(episode_dir, 'images')
depth_dir = os.path.join(episode_dir, 'depth')
csv_path = os.path.join(episode_dir, 'airsim_with_gaze_closest.csv')

# --- LOAD CSV ---
data_rows = []
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        data_rows.append(row)

images = []
depths = []
gazes = []
actions = []
prev_pos_z = None

for row in data_rows:
    img_name = row['ImageFile']
    img_path = os.path.join(images_dir, img_name)
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        continue
    # Load RGB, normalize
    img = Image.open(img_path).convert('RGB').resize((image_size, image_size))
    img_arr = np.asarray(img).astype(np.float32) / 255.0
    images.append(img_arr)

    # Load depth
    base_img_name = img_name.rsplit('.', 1)[0]
    depth_fn = base_img_name + '_depth.png'
    depth_path = os.path.join(depth_dir, depth_fn)
    if not os.path.exists(depth_path):
        print(f"Depth not found: {depth_path}")
        continue
    depth_img = Image.open(depth_path).convert('L').resize((image_size, image_size))
    depth_arr = np.asarray(depth_img).astype(np.float32) / 255.0
    depths.append(depth_arr[..., None])

    # Normalize gaze coordinates (x, y) by image size
    gaze_x = float(row['x']) / 640
    gaze_y = float(row['y']) / 360
    gaze = np.array([[gaze_x], [gaze_y]], dtype=np.float64)
    gazes.append(gaze)

    # Convert quaternion to Euler angles
    q_w = float(row['Q_W'])
    q_x = float(row['Q_X'])
    q_y = float(row['Q_Y'])
    q_z = float(row['Q_Z'])
    roll, pitch, yaw = quaternion_to_euler(q_w, q_x, q_y, q_z)

    # Compute throttle as change in altitude
    pos_z = float(row['POS_Z'])
    if prev_pos_z is not None:
        throttle = pos_z - prev_pos_z
    else:
        throttle = 0.0
    prev_pos_z = pos_z

    action = np.array([[roll], [pitch], [throttle], [yaw]], dtype=np.float64)
    actions.append(action)

N = len(images)
images = np.stack(images, axis=0)
depths = np.stack(depths, axis=0)
gazes = np.stack(gazes, axis=0)
actions = np.stack(actions, axis=0)

print(f"Saving episode {episode}: images {images.shape}, depth {depths.shape}, gaze {gazes.shape}, action {actions.shape}")

np.savez_compressed(
    f"{episode}.npz",
    images=images.astype(np.float32),
    depth=depths.astype(np.float32),
    gaze_coords=gazes.astype(np.float64),
    action=actions.astype(np.float64)
)

print(f'Saved {episode}.npz')
