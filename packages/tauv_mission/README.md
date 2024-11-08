# tauv_mission
This is a package for two things:

 1. Mission-specific code. This means code that is specific to a task, challenge, or otherwise volatile goal that changes with the competition
 2. System-wide launch files. These launch files launch the entire system, including mission, low-level systems, simulator, common packages, etc.

# Launch Files:
## system.launch
This launchfile is configurable to launch the entire system, with the exception of the command system which runs offboard.
**Arguments:**
This launchfile takes a good number of arguments in order to determine how to launch the system:

 - model_name The name of the model to launch. This will determine the namespace used by most systems, as well as which tauv_vehicle launchfile to include.
 - simulated This is a boolean that determines whether to launch the tauv_vehicle package or the simulator
 - simulator_environment The path to the launchfile to launch the simulator environment
 - x,y,z,roll,pitch,yaw These are used in simulated runs to determine the start placement of the robot
 
 **Description:**
 The file is well documented, but the gist of it is that it
  1. Determines the configuration pkg (eg, albatross_description) to use
  2. Finds and loads the vehicle_params file
  3. If simulated, launch the simulator_environment launchfile. Otherwise, launch the \<model_name\>_vehicle.launch file in tauv_vehicle
  4. Upload vehicle URDFs
  5. Launch the thruster manager
  
It might do other things if people add them without updating this documentation >:(
