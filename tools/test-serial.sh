#!/bin/bash

# Serial port (replace with your specific port)
SERIAL_PORT="/dev/maestro"

# Interval in seconds
INTERVAL=1

# Infinite loop
while true; do
    # Write "Hello World" to the serial port
    echo "Hello World" > "$SERIAL_PORT"
    
    # Wait for the specified interval
    sleep "$INTERVAL"
done

