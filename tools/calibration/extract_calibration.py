import depthai as dai

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("ip")
parser.add_argument("output")
args = parser.parse_args()

dev_info = dai.DeviceInfo(args.ip)
pipeline = dai.Pipeline()
dev = dai.Device(pipeline, dev_info)

cal_data = dev.readCalibration()
cal_data.eepromToJsonFile(args.output)

