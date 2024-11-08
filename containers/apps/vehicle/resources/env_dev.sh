#!/bin/bash

export ROS_IP=10.0.0.10
export ROS_MASTER_URI=http://10.0.0.10:11311
roscore >/dev/null &
sleep 3

# Set the session name
SESSION_NAME="TAUV_VEH_DEV"

# Create a new tmux session with a single window
tmux new-session -d -s $SESSION_NAME -n "TELEOP"

# Split the window into 4 equal panes
tmux split-window -h -t $SESSION_NAME:0        # Split window vertically
tmux split-window -v -t $SESSION_NAME:0        # Split left pane horizontally
tmux split-window -h -t $SESSION_NAME:0.2      # Split right pane horizontally

# Split the bottom left pane into 2 (pane 3)
tmux split-window -v -t $SESSION_NAME:0.1


# Send commands to each pane
tmux send-keys -t $SESSION_NAME:0.0 "roslaunch kingfisher_description system.launch"  # Command in top left pane
tmux send-keys -t $SESSION_NAME:0.1 "roslaunch tauv_mission teleop_mission.launch"  # Command in top right pane
tmux send-keys -t $SESSION_NAME:0.3 "rostopic echo /kf/gnc/navigation_state" C-m  # Command in bottom left upper pane
tmux send-keys -t $SESSION_NAME:0.4 "rostopic echo /kf/vehicle/arduino/depth" C-m  # Command in bottom left lower pane
tmux send-keys -t $SESSION_NAME:0.2 "rosbag record -a -o /shared/devel-bag"   # Command in bottom left lower pane


# Attach to the tmux session
tmux attach-session -t $SESSION_NAME
