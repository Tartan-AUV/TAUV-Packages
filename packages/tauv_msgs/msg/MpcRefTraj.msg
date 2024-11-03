# Note: Angular velocities outside of yaw (z axis) are currently unused.

std_msgs/Header header
geometry_msgs/Pose[] poses  # list of poses on trajectory
geometry_msgs/Twist[] twists  # list of twists on trajectory (in world frame! Not body velocities!)
bool auto_twists  # set to True to automatically calculate the twists from the poses, rather than providing them.
bool success  # false indicates some sort of failure