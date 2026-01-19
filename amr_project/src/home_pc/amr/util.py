import math

def compute_distance_and_angle(current_pose, target_pose):
    """
    Computes the Euclidean distance and relative yaw angle from current to target pose.
    args:
     - current_pose: dict with keys 'x', 'y', 'yaw' for the current pose (yaw in radians).
     - target_pose: dict with keys 'x', 'y', 'yaw' for the target pose (yaw in radians).
    return:
     - float: linear distance in meters between the two positions.
     - float: relative heading difference in degrees normalized to [-180, 180].
    """
    # Compute difference
    dx = target_pose['x'] - current_pose['x']
    dy = target_pose['y']  - current_pose['y']

    # Euclidean distance
    distance = math.sqrt(dx**2 + dy**2)

    # Angle in radians and degrees
    # angle_rad = math.atan2(dy, dx)
    diff = target_pose['yaw'] - current_pose['yaw']
    # normalize to range [-pi, pi]
    diff = (diff + math.pi) % (2 * math.pi) - math.pi
    # convert to degrees
    angle_deg = math.degrees(diff)

    return distance, angle_deg

def compute_target_pose(angle, current_pose, distance):
    """
    Computes a target pose after moving a distance at a relative angle from current yaw.
    args:
     - angle: float relative rotation in degrees applied to current yaw.
     - current_pose: dict with keys 'x', 'y', 'yaw' (yaw in radians).
     - distance: float distance to travel in meters.
    return:
     - dict: resulting pose with keys 'x', 'y', 'yaw' (yaw in radians).
    """

    # Convert relative rotation to radians
    relative_angle_rad = math.radians(angle)

    # Global target direction = current yaw + relative rotation
    target_yaw = current_pose['yaw'] + relative_angle_rad

    # Compute new position
    target_x = current_pose['x'] + distance * math.cos(target_yaw)
    target_y = current_pose['y'] + distance * math.sin(target_yaw)

    return {
        'x': target_x,
        'y': target_y,
        'yaw': target_yaw
    }