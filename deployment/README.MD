# Deployment

This folder contains all tools and configuration files for deploying 
code to vehicles. This is also home for all non-containerized infrastructure code
running on the vehicles.

## Deployment locations
### Kingfisher
ROS workspace is located under:
```
/home/tartanauv/tauv_ws
```

Source code:
```
/home/tartanauv/tauv_mono
```

The sim packages listed below do not need to be synchronized to the vehicle:
- `packages/tauv_sim`
- `packages/tauv_sim_gazebo_plugins`
- `packages/tauv_sim_worlds`
- `packages/uuv_simulator`

Following symlinks must be created on the vehicle:
```
ln -s /home/tartanauv/tauv-mono/packages /home/tartanauvshared/tauv_ws/src/tauv
ln -s  /home/tartanauv/tauv-mono/deployment/vehicle/kingfisher/.bashrc /home/tartanauv/.bashrc
```
