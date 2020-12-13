
## Simple use

1. Clone the repository

```
git clone https://github.com/be2rlab/myicub_ros.git
```

2. Build docker image

```
./docker/build_docker.bash
```

3. Run docker container

```
./docker/run_icub.bash
```

4. New *bash* inside docker container

```
./docker/exec_icub.bash
```

and go to the *RUN* part of the readme.


## Dependencies

### Detector

Requirements for Detector:
Cython
matplotlib>=3.2.2
numpy>=1.18.5
opencv-python>=4.1.2
pillow
PyYAML>=5.3
scipy>=1.4.1
tensorboard>=2.2
torch>=1.6.0
torchvision>=0.7.0
tqdm>=4.41.0


### Planner

В папке матлаб запускать необходимо только три скрипта: workspace_and_wall_sd, Config_space, manipulator. Всё остальное - вспомагательные функции.
Запускать в последовательности, указанной выше.
Для отправки траектории в ROS необходимо расскоментировать последние строки в файле manipulator.m.

В папке Catkin лежат файлы для ноды PositionControlMatlab(необходимо создать), которая нужна для получения траектории из матлаба и отправки её на робота в газебо. Файл PositionControlMatlab.h  нужно поместить в PositionControlMatlab/include/PositionControlMatlab, файл PositionControlMatlab.cpp в PositionControlMatlab/src, остальные в PositionControlMatlab

### Как запустить
1. Запустить roscore, yarpserver, matlab.
2. Запустить gazebo и поместить на сцену  iCub.
3. Запустить ноду PositionControlMatlab.
4. Убедиться, что строчки, отвечающие за передачу траектории в ROS в конце файла manipulator.m расскоментированы.
5. Запустить три матлабовских скрипта в последовательности, указанной выше.


```bash
```


## Installation

0. Make new catkin workspace (*DON'T build it now!*)

```
cd ~ && mkdir -p rosicub/src && cd rosicub/src
```

1. Clone icub's gazebo stuff

```
git clone https://github.com/robotology/icub-gazebo.git
```

2. Clone some dependancies

```
git clone https://github.com/ros-planning/geometric_shapes.git -b melodic-devel
```

3. Clone `.sdf` realsense model

*Warn!* Need to modify plugin name in `model.sdf` file!

```
git clone https://github.com/intel/gazebo-realsense.git
```

and inside docker do (this step adds `libRealSensePlugin.so` to gazebo)

```
mkdir build && cd build && cmake .. && make && make install
```

~~Then, make a copy of model~~

```
cp -r ~/rosicub/src/gazebo-realsense/models/realsense_camera ~/rosicub/src/myicub_ros/models/realsense_camera
```

~~and change the plugin~~

```
sed -i "s/filename='libRealSensePlugin.so'/filename='librealsense_gazebo_plugin.so'/g" ~/rosicub/src/myicub_ros/models/realsense_camera/model.sdf
```

4. Clone *ros-gazebo-realsense* plugin

```
git clone https://github.com/pal-robotics/realsense_gazebo_plugin.git -b melodic-devel
```

5.   Clone icub's moveit stuff

<!--
roscore 

 roslaunch launch/realsense_urdf.launch
rosrun gazebo_ros spawn_model -urdf -param robot_description -model rs200
rosrun robot_state_publisher robot_state_publisher

rosrun nodelet nodelet manager screen
rosrun nodelet nodelet load depth_image_proc/convert_metric standalone_nodelet -->

```
git clone https://github.com/bmagyar/icub-moveit.git -b kinetic-devel
```

6. Clone myicub_ros repository

```
git clone https://github.com/be2rlab/myicub_ros.git
```



## RUN

Inside docker
```
yarpserver
```

```
roslaunch myicub_ros start_world.world
```

```
roslaunch myicub_ros depth_proc.launch
```

Joints control ROS interface:
```
rosru myicub_ros myicub_ros
```

and use it

```
rosservice call /state_command ...
```