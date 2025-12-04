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


image_size = 224
base_dir = os.getcwd()

# Collect episode folders (e.g. "1", "2", "episode3", ...)
folders = [f for f in os.listdir(base_dir) if os.path.isdir(f)]
folders = sorted(
    folders,
    key=lambda x: int("".join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else float("inf")
)

for episode in folders:
    episode_dir = os.path.join(base_dir, episode)
    images_dir = os.path.join(episode_dir, "images")
    depth_dir = os.path.join(episode_dir, "depth")
    csv_path = os.path.join(episode_dir, "airsim_with_gaze_closest.csv")

    if not (os.path.exists(images_dir) and os.path.exists(depth_dir) and os.path.exists(csv_path)):
        print(f"Skipping {episode} (missing subfolders or CSV)")
        continue

    # --- LOAD CSV ---
    data_rows = []
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data_rows.append(row)
    except Exception as e:
        print(f"Error reading CSV in {episode}: {e}")
        continue

    images = []
    depths = []
    gazes = []
    actions = []
    prev_pos_z = None

    for i, row in enumerate(data_rows):
        # -------------------------
        # Parse ImageFile field
        # New format: "rgb.png;depth.png"
        # Old format: "img_SimpleFlight__0_....png"
        # -------------------------
        image_field = row["ImageFile"].strip()
        parts = [p.strip() for p in image_field.split(";") if p.strip()]

        if len(parts) == 0:
            print(f"{episode}: [Row {i}] Empty ImageFile field, skipping.")
            continue

        rgb_name = parts[0]

        if len(parts) >= 2:
            depth_name = parts[1]
        else:
            # Backward-compatible depth file naming from old format
            base_img_name = rgb_name.rsplit(".", 1)[0]
            depth_name = base_img_name + "_depth.png"

        rgb_path = os.path.join(images_dir, rgb_name)
        depth_path = os.path.join(depth_dir, depth_name)

        if not os.path.exists(rgb_path):
            print(f"{episode}: [Row {i}] RGB image not found: {rgb_path}")
            continue

        if not os.path.exists(depth_path):
            print(f"{episode}: [Row {i}] Depth image not found: {depth_path}")
            continue

        # -------------------------
        # Load RGB
        # -------------------------
        img = Image.open(rgb_path).convert("RGB").resize((image_size, image_size))
        img_arr = np.asarray(img).astype(np.float32) / 255.0
        images.append(img_arr)

        # -------------------------
        # Load depth
        # -------------------------
        depth_img = Image.open(depth_path).convert("L").resize((image_size, image_size))
        depth_arr = np.asarray(depth_img).astype(np.float32) / 255.0
        depths.append(depth_arr[..., None])

        # -------------------------
        # Gaze normalization
        # -------------------------
        gaze_x = float(row["x"]) / 640.0
        gaze_y = float(row["y"]) / 360.0
        gaze = np.array([[gaze_x], [gaze_y]], dtype=np.float64)
        gazes.append(gaze)

        # -------------------------
        # Orientation and throttle
        # -------------------------
        q_w = float(row["Q_W"])
        q_x = float(row["Q_X"])
        q_y = float(row["Q_Y"])
        q_z = float(row["Q_Z"])
        roll, pitch, yaw = quaternion_to_euler(q_w, q_x, q_y, q_z)

        pos_z = float(row["POS_Z"])
        if prev_pos_z is not None:
            throttle = pos_z - prev_pos_z
        else:
            throttle = 0.0
        prev_pos_z = pos_z

        action = np.array([[roll], [pitch], [throttle], [yaw]], dtype=np.float64)
        actions.append(action)

    # --- SAFETY CHECK BEFORE STACKING ---
    if not images or not depths or not gazes or not actions:
        print(f"{episode}: No valid samples, skipped.")
        continue

    images_np = np.stack(images, axis=0)
    depths_np = np.stack(depths, axis=0)
    gazes_np = np.stack(gazes, axis=0)
    actions_np = np.stack(actions, axis=0)

    print(
        f"Saving {episode}: "
        f"images {images_np.shape}, depth {depths_np.shape}, "
        f"gaze {gazes_np.shape}, action {actions_np.shape}"
    )

    np.savez_compressed(
        f"{episode}.npz",
        images=images_np.astype(np.float32),
        depth=depths_np.astype(np.float32),
        gaze_coords=gazes_np.astype(np.float64),
        action=actions_np.astype(np.float64),
    )
    print(f"Saved {episode}.npz!")
