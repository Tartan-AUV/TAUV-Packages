Header header

int64 ensemble_number

std_msgs/String test_status
std_msgs/String health_status

bool shallow_mode
float64 depth
float64 pressure
float64 pressure_variance

float64 heading
float64 pitch
float64 roll
float64 speed_of_sound
float64 salinity
float64 temperature

float64 transmit_voltage
float64 transmit_current
float64 transmit_impedance

geometry_msgs/Vector3 velocity
float64 velocity_error

geometry_msgs/Vector3 hr_velocity
float64 hr_velocity_error
bool is_hr_velocity_valid

float64[4] beam_ranges
float64[4] beam_correlations
float64[4] beam_amplitudes
float64[4] beam_percent_goods
float64[4] beam_time_to_bottoms
float64[4] beam_standard_deviations
float64[4] beam_time_of_validities
float64[4] beam_rssis

float64 slant_range
float64 axis_delta_range
float64 vertical_range

