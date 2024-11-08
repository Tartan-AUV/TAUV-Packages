import numpy as np
from geometry_msgs.msg import Point

WIDTH_TO_RUN = 20

class DepthEstimator():
    def estimate_relative_depth(depth_image, x, y, bbox):
        box = depth_image[max(bbox.ymin, y - WIDTH_TO_RUN) : min(bbox.ymax, y + WIDTH_TO_RUN+1), max(bbox.xmin, x - WIDTH_TO_RUN) : min(bbox.xmax, x + WIDTH_TO_RUN + 1)]
        return np.nanmean(box) / 1000

    def estimate_absolute_depth(depth_image, bbox, depth_camera_info, known_z=None):
        fx = depth_camera_info.K[0]
        cx = depth_camera_info.K[2]
        fy = depth_camera_info.K[4]
        cy = depth_camera_info.K[5]

        center_x = (bbox.xmin + bbox.xmax) // 2
        center_y = (bbox.ymin + bbox.ymax) // 2
        cur_depth = DepthEstimator.estimate_relative_depth(depth_image, center_x, center_y, bbox)

        if (cur_depth != np.nan):
            cur_x = ((center_x - cx) * cur_depth) / (fx)
            cur_y = ((center_y - cy) * cur_depth) / (fy)

            return (cur_depth, cur_x, cur_y)

        return np.nan