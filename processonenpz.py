import os
import numpy as np
import cv2

npz_path = "101.npz"
out_img_dir = "npz_images"
out_depth_dir = "npz_depth"

os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_depth_dir, exist_ok=True)

data = np.load(npz_path)
images = data["images"]      # (N,224,224,3), [0,1]
depth  = data["depth"]       # (N,224,224,1), [0,1]
action = data["action"]
gaze   = data["gaze_coords"]

print("images:", images.shape, images.dtype)
print("depth :", depth.shape, depth.dtype)
print("action:", action.shape)
print("gaze  :", gaze.shape)

N = images.shape[0]

for i in range(N):
    # RGB image: convert [0,1] float -> uint8 BGR for cv2
    img = (images[i] * 255.0).clip(0, 255).astype("uint8")
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_name = os.path.join(out_img_dir, f"img_{i:05d}.png")
    cv2.imwrite(img_name, img_bgr)

    # depth: single channel
    d = (depth[i, :, :, 0] * 255.0).clip(0, 255).astype("uint8")
    depth_name = os.path.join(out_depth_dir, f"depth_{i:05d}.png")
    cv2.imwrite(depth_name, d)
