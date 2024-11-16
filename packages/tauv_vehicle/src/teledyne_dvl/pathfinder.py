import rospy
import serial
import bitstring
from typing import Optional

from .ensemble import Ensemble

class Pathfinder:

    SERIAL_TIMEOUT = 1
    POLL_TIMEOUT = 1

    MIN_MEASURE_TIME = 0.1
    MAX_MEASURE_TIME = 0.4

    def __init__(self, port: str, baudrate: int):
        self._conn = serial.Serial(port=port, baudrate=baudrate, timeout=Pathfinder.SERIAL_TIMEOUT)
        self._measuring = False

    def open(self):
        self._log('open')

        if not self._conn.isOpen():
            self._conn.open()

    def close(self):
        self._log('close')

        self._conn.close()

    def reset(self):
        self._log('reset')

        self._send_break()

    def configure(self):
        self.reset()
        self._log('configure')

        # Restore factory default settings
        self._send_command('CR1')

        # Set serial parameters
        # 115200bps, no parity, one stop, 8 data
        self._send_command('CB811')

        # Set flow control
        # Automatic ensemble cycling, automatic ping cycling
        # Binary output, serial output
        self._send_command('CF11110')

        # Set trigger enable
        # Ping after low to high transition
        # 0 hundreths of sec delay time
        # Disable timeout
        self._send_command('CX 1 0 65535')

        # Set heading alignment
        # Beam 3 offset by 45 degrees
        self._send_command('EA+04500')

        # Set coordinate transformation
        # Ship coordinates
        # Ignore tilts
        # Allow 3-beam solutions
        # Disable bin mapping
        self._send_command('EX10010')

        # Set sensor source
        self._send_command('EZ11000010')

        # Set salinity
        # 0 ppm
        self._send_command('ES00')

        # Set time per ensemble
        # As fast as possible
        self._send_command('TE00:00:00.00')

        # Set time per ping
        # As fast as possible
        self._send_command('TP00:00.00')

        # Disable water-mass layer
        self._send_command('BK0')

        # Enable single-ping bottom track
        self._send_command('BP001')

        # Set maximum bottom search depth
        # 12m
        self._send_command('BX00120')

        # Set bottom track output types
        # Standard bottom track, high resolution bottom track
        # Precise navigation output
        self._send_command('#BJ100111000')

        # Enable turnkey mode
        # Serial, 5s startup
        self._send_command('CT 1 5')

        # Set output data format
        self._send_command('#PD0')

        # Save settings
        self._send_command('CK')

    def start_measuring(self):
        self._log('start_measuring')

        self._send_command('CS')

        self._measuring = True

    def stop_measuring(self):
        self._log('stop_measuring')

        self._send_break()

        self._measuring = False

    def poll(self) -> Optional[Ensemble]:
        self._log('poll')

        timeout = rospy.Time.now() + rospy.Duration(Pathfinder.POLL_TIMEOUT)

        header_id_1 = self._read(1)
        header_id_2 = self._read(1)

        while (header_id_1 is None
              or header_id_2 is None
              or (header_id_1 + header_id_2).hex() != Ensemble.HEADER_ID):

            if rospy.Time.now() > timeout:
                self._log('poll timeout')
                return None

            header_id_1 = header_id_2
            header_id_2 = self._read(1)

        receive_time = rospy.Time.now()

        e = Ensemble(receive_time=receive_time)

        header_data = header_id_1 + header_id_2 + self._read(Ensemble.HEADER_SIZE - Ensemble.ID_SIZE)
        # self._log('header_data', header_data.hex())

        header = bitstring.BitStream(bytes=header_data)
        
        try:
            packet_size = e.parse_header(header)
        except ValueError as e:
            self._log(e)
            return None

        packet_data = self._read(packet_size)
        # self._log('packet_data', packet_data.hex())

        packet = bitstring.BitStream(bytes=packet_data)

        try:
            e.parse_packet(packet)
        except ValueError as e:
            self._log(e)
            return None

        if not e.is_valid():
            return None

        return e

    def _write(self, data: bytes):
        self._log('[write]', data.hex())
        self._conn.write(data)
        self._conn.flush()

    def _read(self, size: int) -> bytes:
        data = self._conn.read(size)
        # self._log('[read]', data.hex())
        return data

    def _send_break(self):
        self._log('[break]')
        self._conn.send_break(duration=0.3)

    def _send_command(self, cmd: str):
        self._log('[cmd]', cmd)
        cmd += '\r'
        data = cmd.encode('ascii')
        self._write(data)

    def _log(self, *args):
        pass
        # print(*args)
