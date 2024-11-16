# Accelerations
float32 a_x
float32 a_y
float32 a_z
float32 a_roll
float32 a_pitch
float32 a_yaw

# Forces
float32 f_x
float32 f_y
float32 f_z
float32 f_roll
float32 f_pitch
float32 f_yaw

# If set, override accelerations and use forces
bool use_f_x
bool use_f_y
bool use_f_z
bool use_f_roll
bool use_f_pitch
bool use_f_yaw

# Setpoints
float32 setpoint_z
float32 setpoint_roll
float32 setpoint_pitch

# If set, override accelerations and forces and use setpoints
bool use_setpoint_z
bool use_setpoint_roll
bool use_setpoint_pitch