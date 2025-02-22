import rospy
import numpy as np
from typing import Optional

import libm2k

class ADALMChainedBackend:

    def __init__(self, sample_frequency: int, sample_size: int, timeout: float,
                 source_id: str, sink_id: str, source_trig_level: float, sink_trig_level: float):
        self._sample_frequency: int = sample_frequency
        self._sample_size: int = sample_size
        self._timeout: int = int(timeout / 1000)

        self._source_id: str = source_id
        self._sink_id: str = sink_id

        self._source_trig_level: float = source_trig_level
        self._sink_trig_level: float = sink_trig_level

        self._ctx0: Optional[libm2k.Context] = None
        self._ctx1: Optional[libm2k.Context] = None
        self._ain0: Optional[libm2k.M2kAnalogIn] = None
        self._ain1: Optional[libm2k.M2kAnalogIn] = None
        self._trig0: Optional[libm2k.M2kHardwareTrigger] = None
        self._trig1: Optional[libm2k.M2kHardwareTrigger] = None

    def open(self) -> bool:
        ctxs = libm2k.getAllContexts()

        if len(ctxs) != 2:
            rospy.logerr(f'Could not find 2 contexts. Available contexts: {ctxs}')
            return False

        ctx0 = None
        ctx1 = None

        try:
            ctx0 = libm2k.m2kOpen(ctxs[0])
            ctx1 = libm2k.m2kOpen(ctxs[1])
        except ValueError as e:
            rospy.logerr(f'Open failed: {e}')
            return False

        if ctx0.getSerialNumber() == self._source_id and ctx1.getSerialNumber() == self._sink_id:
            self._ctx0 = ctx0
            self._ctx1 = ctx1
        elif ctx0.getSerialNumber() == self._sink_id and ctx1.getSerialNumber() == self._source_id:
            self._ctx0 = ctx1
            self._ctx1 = ctx0
        else:
            rospy.logerr(f'Could not match context serial numbers. Available contexts: {ctxs}')
            return False

        self._ain0, self._trig0 = self._configure_ctx(self._ctx0)
        self._ain1, self._trig1 = self._configure_ctx(self._ctx1)

        self._configure_source_trig(self._trig0)
        self._configure_sink_trig(self._trig1)

        return True

    def close(self):
        if self._ctx0 is not None:
            libm2k.contextClose(self._ctx0)
            self._ctx0 = None
            self._ain0 = None
            self._trig0 = None

        if self._ctx1 is not None:
            libm2k.contextClose(self._ctx1)
            self._ctx1 = None
            self._ain1 = None
            self._trig1 = None


    def sample(self) -> (bool, Optional[np.array], Optional[np.array]):
        rospy.loginfo('sampling')

        self._ain0.startAcquisition(self._sample_size)
        self._ain1.startAcquisition(self._sample_size)

        rospy.loginfo('acquisition started')

        data0 = None
        data1 = None

        try:
            data0 = self._ain0.getSamples(self._sample_size)
            data1 = self._ain1.getSamples(self._sample_size)
        except:
            rospy.loginfo('Sample timeout.')

        rospy.loginfo('got samples')

        self._ain0.stopAcquisition()
        self._ain1.stopAcquisition()

        rospy.loginfo('stopped acquisition')

        if data0 is None or data1 is None:
            return False, None, None

        times = np.tile(np.arange(0, self._sample_size) / self._sample_frequency, (4, 1))
        samples = np.array([data0[0], data0[1], data1[0], data1[1]])

        rospy.loginfo('acquisition finished')

        return True, times, samples


    def set_timeout(self, timeout: int):
        self._timeout = timeout

        if self._ctx0 is not None:
            self._ctx0.set_timeout(self._timeout)

        if self._ctx1 is not None:
            self._ctx1.set_timeout(self._timeout)


    def _configure_ctx(self, ctx: libm2k.Context) -> (libm2k.M2kAnalogIn, libm2k.M2kHardwareTrigger):
        ctx.setTimeout(self._timeout)

        ain = ctx.getAnalogIn()
        trig = ain.getTrigger()

        ain.reset()

        ain.enableChannel(0, True)
        ain.enableChannel(1, True)
        ain.setSampleRate(self._sample_frequency)
        ain.setRange(0, libm2k.PLUS_MINUS_2_5V)
        ain.setRange(1, libm2k.PLUS_MINUS_2_5V)

        ctx.calibrateADC()

        return ain, trig


    def _configure_source_trig(self, trig: libm2k.M2kHardwareTrigger):
        trig.setAnalogMode(0, libm2k.ANALOG)
        trig.setAnalogExternalOutSelect(libm2k.SELECT_ANALOG_IN)
        trig.setAnalogSource(0)
        trig.setAnalogCondition(0, libm2k.RISING_EDGE_ANALOG)
        trig.setAnalogLevel(0, self._source_trig_level)
        trig.setAnalogDelay(0)


    def _configure_sink_trig(self, trig: libm2k.M2kHardwareTrigger):
        trig.setAnalogMode(0, libm2k.EXTERNAL)
        trig.setAnalogSource(0)
        trig.setAnalogExternalCondition(0, libm2k.RISING_EDGE_ANALOG)
        trig.setAnalogLevel(0, self._sink_trig_level)
        trig.setAnalogDelay(0)
