#!/usr/bin/env bash

# xhost +local:

#!!! setup it to use ROS
# HOST_IP=192.168.1.182

xhost +local:docker || true

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/" && pwd )"

if [[ $1 = "--nvidia" ]] || [[ $1 = "-n" ]]
  then
    docker run --gpus all \
                -ti --rm \
                -e "DISPLAY" \
                -e "QT_X11_NO_MITSHM=1" \
                -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
                -e XAUTHORITY \
                -v /dev:/dev \
                -v /home/$USER/icub_ws/icub-grasping/:/icub-grasping \
                -v /home/$USER/:/user \
               --net=host \
               --privileged \
               --name myicub myicub

else

    echo "[!] If you wanna use nvidia gpu, please use script with -n or --nvidia argument"
    docker run  -ti --rm \
                -e "DISPLAY" \
                -e "QT_X11_NO_MITSHM=1" \
                -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
                -e XAUTHORITY \
                -v /dev:/dev \
                -v /home/$USER/icub_ws/icub-grasping/:/icub-grasping \
                -v /home/$USER/matlab/:/matlab \
                -v /home/$USER/:/user \
                -e Matlab_ROOT_DIR=/matlab \
               --net=host \
               --privileged \
               --name myicub myicub
fi
