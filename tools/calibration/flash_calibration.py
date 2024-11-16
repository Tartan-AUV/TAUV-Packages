import depthai as dai

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('filename')
parser.add_argument('ip')
args = parser.parse_args()

devinfo = dai.DeviceInfo(args.ip)
pipeline = dai.Pipeline()
device = dai.Device(pipeline, devinfo)
with open(args.filename, 'r') as file:
    cal_json = json.load(file)
cal_handler = dai.CalibrationHandler.fromJson(cal_json)
device.flashCalibration(cal_handler)
