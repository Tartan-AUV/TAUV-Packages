import numpy as np
import RPi.GPIO as gpio


class AS1Rev4Frontend:

    def __init__(self, gain_pins: [int]):
        assert(len(gain_pins) == 3)

        self._gain_pins = gain_pins

    def open(self):
        gpio.setmode(gpio.BCM)
        gpio.setup(self._gain_pins[0], gpio.OUT, initial=gpio.LOW)
        gpio.setup(self._gain_pins[1], gpio.OUT, initial=gpio.LOW)
        gpio.setup(self._gain_pins[2], gpio.OUT, initial=gpio.LOW)

    def close(self):
        gpio.cleanup()

    def set_gain(self, gain):
        assert(1 <= gain <= 8)
        gain_index = gain - 1

        bits = [bool(int(bit)) for bit in format(gain_index, '03b')]

        gpio.output(self._gain_pins[2], bits[0])
        gpio.output(self._gain_pins[1], bits[1])
        gpio.output(self._gain_pins[0], bits[2])

    def min_gain(self) -> int:
        return 1

    def max_gain(self) -> int:
        return 8