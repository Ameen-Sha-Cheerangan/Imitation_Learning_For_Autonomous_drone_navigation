import airsim
import time
import numpy as np

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

# Arm and takeoff
print("Arming and taking off...")
client.armDisarm(True)
client.takeoffAsync().join()
client.hoverAsync().join()

# Basic movement commands
def move_drone():
    # Move forward for 3 seconds
    print("Moving forward...")
    client.moveByVelocityAsync(5, 0, 0, 3).join()

    # Turn right (yaw)
    print("Turning right...")
    client.rotateByYawRateAsync(30, 2).join()

    # Move to specific position
    print("Moving to target position...")
    client.moveToPositionAsync(10, 10, -5, 5).join()

    # Hover for 2 seconds
    print("Hovering...")
    client.hoverAsync().join()
    time.sleep(2)

# Execute movements
move_drone()

# Land and disarm
print("Landing...")
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)


# ./LandscapeMountains.sh -ResX=640 -ResY=480 -windowed -BENCHMARK -fps=30
