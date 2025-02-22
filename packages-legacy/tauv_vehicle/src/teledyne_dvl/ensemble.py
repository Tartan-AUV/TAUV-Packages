import rospy
import bitstring

from geometry_msgs.msg import Vector3
from std_msgs.msg import Header, String
from tauv_msgs.msg import TeledyneDvlData as DvlDataMsg

class Ensemble:

    TEST_STATUS_MAPPING = {
        '01': 'Transmitter shutdown',
        '02': 'Transmitter overcurrent',
        '03': 'Transmitter undercurrent',
        '04': 'Transmitter undervoltage',
        '10': 'FIFO interrupt missed',
        '11': 'FIFO ISR re-entry',
        '21': 'Sensor start failure',
        '22': 'Temperature sensor failure',
        '23': 'Pressure sensor failure',
        '27': 'Bad comms with sensor',
        '28': 'Bad comms with sensor',
        '29': 'Sensor cal data checksum failure',
        '30': 'Stuck UART',
        '31': 'QUART transmit timeout',
        '32': 'QUART IRQ stuck',
        '33': 'QUART buffer stuck',
        '34': 'QUART IRQ active',
        '35': 'QUART cannot clear interrupt',
        '50': 'RTC low battery',
        '60': 'Lost nonvolatile pointers',
        '61': 'Erase operation failed',
        '62': 'Error writing from flash to buffer 1',
        '63': 'Error writing from buffer 1 to flash',
        '64': 'Timed out checking if page is erased',
        '65': 'Bad return when checking page',
        '66': 'Loop recorder slate full',
        '70': 'Unable to write to FRAM',
        '80': 'HEM data corrupt or not initialized',
        '81': 'HEM data corrupt or not initialized',
        '82': 'Failed to update HEM data',
        '83': 'Failed to update HEM data',
        '84': 'Failed to read HEM time data',
        '85': 'Failed to read HEM pressure data',
        '86': 'Failed to read HEM SPI state',
        '87': 'Operating time over max',
        '88': 'Pressure reading over sensor limit',
        '89': 'Leak detected in sensor A',
        '8A': 'Leak detected in sensor B',
        'FF': 'Power failure',
    }

    HEALTH_STATUS_MAPPING = {
        0b00000001: 'Leak sensor A leak detected',
        0b00000010: 'Leak sensor A open circuit',
        0b00000100: 'Leak sensor B leak detected',
        0b00001000: 'Leak sensor B open circuit',
        0b00010000: 'Tx voltage updated',
        0b00100000: 'Tx current updated',
        0b01000000: 'Transducer impedance updated',
    }

    ID_SIZE = 2
    HEADER_SIZE = 6

    DATATYPE_OFFSET = 6

    CHECKSUM_SIZE = 2
    CHECKSUM_MODULO = 65536

    HEADER_ID = '7f7f'
    FIXED_LEADER_ID = '0000'
    VARIABLE_LEADER_ID = '8000'
    BOTTOM_TRACK_DATA_ID = '0006'
    HR_BOTTOM_TRACK_DATA_ID = '0358'
    NAV_PARAMS_DATA_ID = '1320'
    BOTTOM_TRACK_RANGE_DATA_ID = '0458'

    FIXED_LEADER_SIZE = 58
    VARIABLE_LEADER_SIZE = 77
    BOTTOM_TRACK_DATA_SIZE = 81
    HR_BOTTOM_TRACK_DATA_SIZE = 70
    NAV_PARAMS_DATA_SIZE = 85
    BOTTOM_TRACK_RANGE_DATA_SIZE = 41

    def __init__(self, receive_time: rospy.Time):
        self.receive_time = receive_time
        self.parsed_fixed_leader = False
        self.parsed_variable_leader = False
        self.parsed_bottom_track_data = False
        self.parsed_hr_bottom_track_data = False
        self.parsed_nav_params_data = False
        self.parsed_bottom_track_range_data = False

    def parse_header(self, d: bitstring.BitStream) -> int:
        self.header_data = d.bytes

        header_id = d.read('hex:16')
        if header_id != Ensemble.HEADER_ID:
            raise ValueError('Unexpected Header ID: {}'.format(header_id))

        self.ensemble_size = d.read('uintle:16')

        d.bytepos += 1

        self.datatype_count = d.read('uint:8')

        self.packet_size = self.ensemble_size - Ensemble.HEADER_SIZE + Ensemble.CHECKSUM_SIZE

        return self.packet_size

    def parse_packet(self, d: bitstring.BitStream):
        self.packet_data = d.bytes

        self._parse_datatype_offsets(d)

        for datatype_offset in self.datatype_offsets:
            self._parse_datatype(d, datatype_offset)

        self._validate_checksum(d)

    def is_valid(self) -> bool:
        if not (self.parsed_fixed_leader
                and self.parsed_variable_leader
                and self.parsed_bottom_track_data
                and self.parsed_hr_bottom_track_data
                and self.parsed_nav_params_data
                and self.parsed_bottom_track_range_data):
            return False

        return True

    def to_msg(self) -> DvlDataMsg:
        msg = DvlDataMsg()
        msg.header = Header()

        msg.header.stamp = rospy.get_rostime()

        if self.parsed_variable_leader:
            msg.ensemble_number = (self.ensemble_number_msb << 16) + self.ensemble_number
            msg.test_status.data = Ensemble.TEST_STATUS_MAPPING.get(self.test_status, 'Unknown error')
            msg.health_status.data = self._get_health_status(self.health_status)
            msg.depth = self.depth * 1e-1
            msg.pressure = (self.pressure / 10132.5) + 1 # Convert decapascals to ATM
            msg.pressure_variance = (self.pressure_variance / 10132.5) + 1 # Convert decapascals to ATM
            msg.heading = self.heading * 1e-2
            msg.pitch = self.pitch * 1e-2
            msg.roll = self.roll * 1e-2
            msg.speed_of_sound = self.speed_of_sound * 1e0
            msg.salinity = self.salinity * 1e0
            msg.temperature = self.temperature * 1e-2
            msg.transmit_voltage = self.transmit_voltage * 1e-3
            msg.transmit_current = self.transmit_current * 1e-3
            msg.transmit_impedance = self.transmit_impedance * 1e-3

        if self.parsed_bottom_track_data:
            msg.velocity = Vector3(
                0 if -self.velocity_y == 0x8000 else self.velocity_y * -1e-3,
                0 if -self.velocity_x == 0x8000 else self.velocity_x * -1e-3,
                0 if -self.velocity_z == 0x8000 else self.velocity_z * -1e-3
            )
            msg.velocity_error = 0 if -self.velocity_error == 0x8000 else self.velocity_error * 1e-3

            msg.beam_ranges = [
                ((self.beam_1_range_msb << 16) | self.beam_1_range) * 1e0,
                ((self.beam_2_range_msb << 16) | self.beam_2_range) * 1e0,
                ((self.beam_3_range_msb << 16) | self.beam_3_range) * 1e0,
                ((self.beam_4_range_msb << 16) | self.beam_4_range) * 1e0,
            ]
            msg.beam_correlations = [
                self.beam_1_correlation * 1e0,
                self.beam_2_correlation * 1e0,
                self.beam_3_correlation * 1e0,
                self.beam_4_correlation * 1e0,
            ]
            msg.beam_amplitudes = [
                self.beam_1_amplitude * 1e0,
                self.beam_2_amplitude * 1e0,
                self.beam_3_amplitude * 1e0,
                self.beam_4_amplitude * 1e0,
            ]
            msg.beam_percent_goods = [
                self.beam_1_percent_good * 1e0,
                self.beam_2_percent_good * 1e0,
                self.beam_3_percent_good * 1e0,
                self.beam_4_percent_good * 1e0,
            ]
            msg.beam_rssis = [
                self.beam_1_rssi * 1e0,
                self.beam_2_rssi * 1e0,
                self.beam_3_rssi * 1e0,
                self.beam_4_rssi * 1e0,
            ]

        if self.parsed_hr_bottom_track_data:
            msg.hr_velocity = Vector3(
                0 if -self.hr_velocity_x == 0x80000000 else self.hr_velocity_x * 1e-5,
                0 if -self.hr_velocity_y == 0x80000000 else self.hr_velocity_y * 1e-5,
                0 if -self.hr_velocity_z == 0x80000000 else self.hr_velocity_z * 1e-5,
            )
            msg.hr_velocity_error = 0 if -self.hr_velocity_error == 0x80000000 else self.hr_velocity_error * 1e-5
            msg.is_hr_velocity_valid = -self.hr_velocity_x != 0x80000000 and \
                                       -self.hr_velocity_y != 0x80000000 and \
                                       -self.hr_velocity_z != 0x80000000

        if self.parsed_nav_params_data:
            msg.shallow_mode = bool(self.shallow_mode)
            msg.beam_time_to_bottoms = [
                self.beam_1_time_to_bottom * 13.02e-6,
                self.beam_2_time_to_bottom * 13.02e-6,
                self.beam_3_time_to_bottom * 13.02e-6,
                self.beam_4_time_to_bottom * 13.02e-6,
            ]
            msg.beam_standard_deviations = [
                self.beam_1_standard_deviation * 1e-3,
                self.beam_2_standard_deviation * 1e-3,
                self.beam_3_standard_deviation * 1e-3,
                self.beam_4_standard_deviation * 1e-3,
            ]
            msg.beam_time_of_validities = [
                self.beam_1_time_of_validity * 1e-6,
                self.beam_2_time_of_validity * 1e-6,
                self.beam_3_time_of_validity * 1e-6,
                self.beam_4_time_of_validity * 1e-6,
            ]

        if self.parsed_bottom_track_range_data:
            msg.slant_range = self.slant_range * 1e-4
            msg.axis_delta_range = self.axis_delta_range * 1e-4
            msg.vertical_range = self.vertical_range * 1e-4

        return msg

    def _parse_datatype_offsets(self, d: bitstring.BitStream):
        d.bytepos = 0
        self.datatype_offsets = []

        for i in range(self.datatype_count):
            datatype_offset = d.read('uintle:16') - Ensemble.DATATYPE_OFFSET
            self.datatype_offsets.append(datatype_offset)

    def _parse_datatype(self, d: bitstring.BitStream, datatype_offset: int):
        d.bytepos = datatype_offset
        datatype_id = d.read('hex:16')
        d.bytepos = datatype_offset

        rospy.loginfo('Parsing {} at {}'.format(datatype_id, datatype_offset))

        if datatype_id == Ensemble.FIXED_LEADER_ID:
            self._parse_fixed_leader(d[8*d.bytepos:])
            self.parsed_fixed_leader = True
        elif datatype_id == Ensemble.VARIABLE_LEADER_ID:
            self._parse_variable_leader(d[8*d.bytepos:])
            self.parsed_variable_leader = True
        elif datatype_id == Ensemble.BOTTOM_TRACK_DATA_ID:
            self._parse_bottom_track_data(d[8*d.bytepos:])
            self.parsed_bottom_track_data = True
        elif datatype_id == Ensemble.HR_BOTTOM_TRACK_DATA_ID:
            self._parse_hr_bottom_track_data(d[8*d.bytepos:])
            self.parsed_hr_bottom_track_data = True
        elif datatype_id == Ensemble.NAV_PARAMS_DATA_ID:
            self._parse_nav_params_data(d[8*d.bytepos:])
            self.parsed_nav_params_data = True
        elif datatype_id == Ensemble.BOTTOM_TRACK_RANGE_DATA_ID:
            self._parse_bottom_track_range_data(d[8*d.bytepos:])
            self.parsed_bottom_track_range_data = True
        else:
            rospy.logwarn('Unexpected datatype ID: {}'.format(datatype_id))

    def _parse_fixed_leader(self, d: bitstring.BitStream):
        pass

    def _parse_variable_leader(self, d: bitstring.BitStream):
        d.bytepos = 2
        self.ensemble_number = d.read('uintle:16')

        d.bytepos = 11
        self.ensemble_number_msb = d.read('uintle:8')
        self.test_status = d.read('hex:8')

        d.bytepos = 14
        self.speed_of_sound = d.read('uintle:16')
        self.depth = d.read('uintle:16')
        self.heading = d.read('uintle:16')
        self.pitch = d.read('uintle:16')
        self.roll = d.read('uintle:16')
        self.salinity = d.read('uintle:16')
        self.temperature = d.read('intle:16')

        d.bytepos = 48
        self.pressure = d.read('uintle:32')
        self.pressure_variance = d.read('uintle:32')

        d.bytepos = 66
        self.health_status = d.read('uintle:8')

        d.bytepos = 73
        self.transmit_voltage = d.read('uintle:16')
        self.transmit_current = d.read('uintle:16')
        self.transmit_impedance = d.read('uintle:16')

    def _parse_bottom_track_data(self, d: bitstring.BitStream):
        d.bytepos = 16
        self.beam_1_range = d.read('uintle:16')
        self.beam_2_range = d.read('uintle:16')
        self.beam_3_range = d.read('uintle:16')
        self.beam_4_range = d.read('uintle:16')
        self.velocity_x = d.read('intle:16')
        self.velocity_y = d.read('intle:16')
        self.velocity_z = d.read('intle:16')
        self.velocity_error = d.read('intle:16')
        self.beam_1_correlation = d.read('uintle:8')
        self.beam_2_correlation = d.read('uintle:8')
        self.beam_3_correlation = d.read('uintle:8')
        self.beam_4_correlation = d.read('uintle:8')
        self.beam_1_amplitude = d.read('uintle:8')
        self.beam_2_amplitude = d.read('uintle:8')
        self.beam_3_amplitude = d.read('uintle:8')
        self.beam_4_amplitude = d.read('uintle:8')
        self.beam_1_percent_good = d.read('uintle:8')
        self.beam_2_percent_good = d.read('uintle:8')
        self.beam_3_percent_good = d.read('uintle:8')
        self.beam_4_percent_good = d.read('uintle:8')

        d.bytepos = 72
        self.beam_1_rssi = d.read('uintle:8')
        self.beam_2_rssi = d.read('uintle:8')
        self.beam_3_rssi = d.read('uintle:8')
        self.beam_4_rssi = d.read('uintle:8')

        d.bytepos = 77
        self.beam_1_range_msb = d.read('uintle:8')
        self.beam_2_range_msb = d.read('uintle:8')
        self.beam_3_range_msb = d.read('uintle:8')
        self.beam_4_range_msb = d.read('uintle:8')

    def _parse_hr_bottom_track_data(self, d: bitstring.BitStream):
        d.bytepos = 2
        self.hr_velocity_x = d.read('intle:32')
        self.hr_velocity_y = d.read('intle:32')
        self.hr_velocity_z = d.read('intle:32')
        self.hr_velocity_error = d.read('intle:32')

    def _parse_nav_params_data(self, d: bitstring.BitStream):
        d.bytepos = 2
        self.beam_1_time_to_bottom = d.read('uintle:32')
        self.beam_2_time_to_bottom = d.read('uintle:32')
        self.beam_3_time_to_bottom = d.read('uintle:32')
        self.beam_4_time_to_bottom = d.read('uintle:32')

        self.beam_1_standard_deviation = d.read('uintle:16')
        self.beam_2_standard_deviation = d.read('uintle:16')
        self.beam_3_standard_deviation = d.read('uintle:16')
        self.beam_4_standard_deviation = d.read('uintle:16')

        self.shallow_mode = d.read('uintle:8')

        d.bytepos = 53
        self.beam_1_time_of_validity = d.read('uintle:32')
        self.beam_2_time_of_validity = d.read('uintle:32')
        self.beam_3_time_of_validity = d.read('uintle:32')
        self.beam_4_time_of_validity = d.read('uintle:32')

    def _parse_bottom_track_range_data(self, d: bitstring.BitStream):
        d.bytepos = 2
        self.slant_range = d.read('uintle:32')
        self.axis_delta_range = d.read('uintle:32')
        self.vertical_range = d.read('uintle:32')

    def _validate_checksum(self, d: bitstring.BitStream):
        d.bytepos = self.packet_size - Ensemble.CHECKSUM_SIZE

        self.expected_checksum = d.read('uintle:16')

        data_to_sum = self.header_data + self.packet_data[:-Ensemble.CHECKSUM_SIZE]

        self.checksum = sum(data_to_sum) % Ensemble.CHECKSUM_MODULO

        if self.checksum != self.expected_checksum:
            raise ValueError('Unexpected Checksum')

    def _get_health_status(self, i: int) -> str:
        status = ''

        for (id, message) in Ensemble.HEALTH_STATUS_MAPPING.items():
            if i & id != 0:
                if len(status) > 0:
                    status += '; '
                status += message

        return status
