#!/usr/bin/env bash
# source rosrerun.sh my_partial_node_name
# https://gist.github.com/lucasw/292dc8d7dc82d6c4e99350498e173181

node_name=$1
echo $node_name

# Get the command to launch the node, which should have __name
# and remaps, but args are still on the parameter server so
# are okay not to be here.
# Only take first match, which may fail in combination
# with the rosnode list below
orig_cmd=`ps -eo args | grep  "__name:=$node_name" | grep __name | head -n 1`
# get the namespace
# (why don't roslaunched nodes have namespaces in the ps output?)
node_path=`rosnode list | grep $node_name$ | sed 's:/[^/]*$::'`
# adding the namespace here may be redundant but is harmless
# if specified twice
cmd="$orig_cmd __ns:=$node_path"
echo "press ctrl-c then up arrow to run this again (only works if running this with source)"
history -s $cmd
$cmd