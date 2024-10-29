#!/bin/bash

rosrun kalibr kalibr_calibrate_cameras \
  --bag /data/oakd_bottom.bag --target /data/target.yaml \
  --models pinhole-radtan pinhole-radtan pinhole-radtan --topics /kf/vehicle/oakd_bottom/color/image_raw /kf/vehicle/oakd_bottom/right/image_raw /kf/vehicle/oakd_bottom/left/image_raw
