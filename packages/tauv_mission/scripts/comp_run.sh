#!/bin/bash
roslaunch kingfisher_description system.launch &
sleep 15

roslaunch tauv_mission mission_manager.launch &
sleep 5

rosbag record -a -o /shared/comp_run.bag &

rosservice call /kf/mission/

#!/bin/bash

roscore >/dev/null &
export ROS_IP=10.0.0.10
export ROS_MASTER_URI=http://10.0.0.10:11311

# Set the session name
MISSION_NAME="kf_buoy_dive_24"

SESSION_NAME="TAUV_COMP"

# Create a new tmux session with a single window
tmux new-session -d -s $SESSION_NAME -n "COMP"

# Split the window into 4 equal panes
tmux split-window -h -t $SESSION_NAME:0         # Split window vertically
tmux split-window -v -t $SESSION_NAME:0        # Split left pane horizontally
tmux split-window -v -t $SESSION_NAME:0.2      # Split right pane horizontally

# Split the bottom left pane into 2 (pane 3)
tmux split-window -v -t $SESSION_NAME:0.1
tmux split-window -v -t $SESSION_NAME:0.1

tmux split-window -v -t $SESSION_NAME:0.2

# Send commands to each pane
tmux send-keys -t $SESSION_NAME:0.0 "roslaunch kingfisher_description system.launch" C-m # Command in top left pane
sleep 15
tmux send-keys -t $SESSION_NAME:0.2 "roslaunch tauv_mission mission_manager.launch" C-m  # Command in top right pane
sleep 5
tmux send-keys -t $SESSION_NAME:0.3 "rosbag record -a -o /shared/devel-bag" C-m   # Command in bottom left lower pane
tmux send-keys -t $SESSION_NAME:0.1 "rostopic echo /kf/gnc/navigation_state" C-m  # Command in bottom left upper pane
tmux send-keys -t $SESSION_NAME:0.3 "rostopic echo /kf/vehicle/arduino/depth" C-m  # Command in bottom left lower pane
sleep 5
tmux send-keys -t $SESSION_NAME:0.3 "rosservice call /kf/mission/run $(MISSION_NAME)" C-m  # Command in bottom left lower pane
# Attach to the tmux session
tmux attach-session -t $SESSION_NAME
