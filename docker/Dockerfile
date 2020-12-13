ARG from
FROM ${from}
# FROM robotology/robotology-tdd
LABEL maintainer="kaartemov@gmail.com"

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    apt-utils \
    net-tools \
    mesa-utils \
    gnupg2 \
    wget \
    curl \
    git \
    mc \
    nano \
    cmake \
    cmake-curses-gui \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Timezone Configuration
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu bionic main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -sSL 'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC1CF6E31E6BADE8868B172B4F42ED6FBAB17C654' | apt-key add -
RUN apt-get update && apt-get install -y ros-melodic-ros-base

# yarp for ubuntu 18.04 https://github.com/robotology/QA/issues/364
RUN apt-get update && apt-get install -y apt-transport-https ca-certificates gnupg software-properties-common wget
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
RUN apt-get update && apt-get install -y kitware-archive-keyring
RUN apt-key --keyring /etc/apt/trusted.gpg del C1F34CDD40CD72DA
RUN apt-get install -y cmake cmake-curses-gui

# install https://github.com/robotology/ycm/
# from binaries (in university network didn't work)
# RUN sh -c 'echo "deb http://www.icub.org/ubuntu focal contrib/science" > /etc/apt/sources.list.d/icub.list'
# RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 57A5ACB6110576A6
# RUN apt-get update && apt-get install -y ycm-cmake-modules

# from source
RUN git clone https://github.com/robotology/ycm.git
RUN cd ycm && mkdir build && cd build && cmake ../ && make && make install


# install ALL dependencies for ALL features
# TODO need to clear it
RUN apt-get update && apt-get install -y build-essential git cmake cmake-curses-gui \
            libeigen3-dev \
            libace-dev \
            libedit-dev \
            libsqlite3-dev \
            libtinyxml-dev \
            qtbase5-dev qtdeclarative5-dev qtmultimedia5-dev \
            qml-module-qtquick2 qml-module-qtquick-window2 \
            qml-module-qtmultimedia qml-module-qtquick-dialogs \
            qml-module-qtquick-controls qml-module-qt-labs-folderlistmodel \
            qml-module-qt-labs-settings \
            libqcustomplot-dev \
            libgraphviz-dev \
            libjpeg-dev \
            libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
            gstreamer1.0-plugins-base \
            gstreamer1.0-plugins-good \
            gstreamer1.0-plugins-bad \
            gstreamer1.0-libav

RUN apt-get update && apt-get install -y \
         coinor-libipopt-dev freeglut3-dev libboost-system-dev \
         libboost-filesystem-dev libboost-thread-dev  \
         libgsl0-dev libopencv-dev libode-dev \
         liblua5.1-dev lua5.1 git swig \
         libsdl1.2-dev \
         libxml2-dev libv4l-dev

RUN apt-get update && apt-get install --no-install-recommends -y \
    libgazebo* gazebo* \
  && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y caffe-cpu libmatio-dev
# RUN sudo apt install -y caffe-cuda

# need to make robotology-superbuild
RUN git config --global user.name human && git config --global user.email human@gmail.com

# better do it using volume
# RUN git clone https://github.com/robotology/robotology-superbuild.git
# RUN cd robotology-superbuild && mkdir build && cd build && ccmake ../ && make

# install dart library
RUN apt-add-repository ppa:dartsim/ppa && apt-get update && apt-get install -y libdart6-all-dev libtinyxml2-dev liburdfdom-dev \
    libxi-dev libxmu-dev freeglut3-dev libopenscenegraph-dev 

RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install tensorflow
# RUN pip3 install opencv-python
RUN pip3 install object_detection
RUN pip3 install tf-object-detection
RUN pip3 install dnn

RUN apt-get update && apt-get install -y python-pip
RUN apt-get update && apt-get install -y python-catkin-tools

RUN python3 -m pip install -U pip
RUN python3 -m pip install -U setuptools
RUN pip3 install opencv-python
RUN pip3 install pyyaml
RUN pip3 install scipy
RUN pip3 install tensorboard


RUN apt-get update && apt-get install -y ros-melodic-eigen-stl-containers
RUN apt-get update && apt-get install -y ros-melodic-control*
RUN apt-get update && apt-get install -y ros-melodic-random-numbers ros-melodic-resource-retriever \
        ros-melodic-shape-msgs ros-melodic-visualization-msgs

RUN apt-get update && apt-get install -y ros-melodic-moveit  ros-melodic-joint* ros-melodic-robot-state-publisher

RUN apt-get update && apt-get install -y ros-melodic-gazebo-ros
RUN apt-get update && apt-get install -y ros-melodic-camera-info-manager
RUN apt-get update && apt-get install -y ros-melodic-depth-image-proc


RUN apt-get install -y ros-melodic-visp* \
                       ros-melodic-camera-info-manager* \
                       ros-melodic-image-transport* \
                       ros-melodic-codec-image-transport \
                       ffmpeg

RUN apt-get install -y ros-melodic-usb-cam ros-melodic-image-view

COPY config.yaml /root/.ignition/fuel/config.yaml

RUN echo "source /icub-grasping/robotology-superbuild/build/install/share/robotology-superbuild/setup.sh" >> /root/.bashrc
# RUN echo "source /opt/ros/melodic/setup.bash" >> /root/.bashrc
RUN echo "source /user/rosicub/devel/setup.bash" >> /root/.bashrc

RUN echo 'export GAZEBO_MODEL_PATH=${GAZEBO_MODEL_PATH}:/user/rosicub/src/myicub_ros/models' >> /root/.bashrc
RUN echo 'export GAZEBO_PLUGIN_PATH=${GAZEBO_PLUGIN_PATH}:/user/rosicub/devel/lib' >> /root/.bashrc

# COPY ./env_setup.bash /env_setup.bash
#ENTRYPOINT [ "env_setup.bash" ]
