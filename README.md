## Installation

1. Clone the repository (*DON'T build it now!*) 

```
mkdir -p ~/rosicub/src && cd ~/rosicub/src && git clone https://github.com/be2rlab/myicub_ros.git
```

2. Build docker image

```
./docker/build_docker.bash
```

### Check if docker's stuff works well

* Run docker container

```
./docker/run_icub.bash
```

* New *bash* inside docker container

```
./docker/exec_icub.bash
```

3. **Do it on the host.** Check or make new robotology superbuild path

```
mkdir -p ~/icub_ws/icub-grasping
```

```
git clone https://github.com/robotology/robotology-superbuild.git
```

**Inside docker!** Build robotology superbuild

```
cd /icub-grasping/robotology-superbuild && mkdir build && cd build && cmake .. && make -j16
```

**All next steps on the host!**

```
cd ~/rosicub/src
```

```
git clone https://github.com/robotology/icub-gazebo.git
```

```
git clone https://github.com/ros-planning/geometric_shapes.git -b melodic-devel
```

```
git clone https://github.com/intel/gazebo-realsense.git
```

**The step inside docker** and inside docker do (adds `libRealSensePlugin.so` to gazebo)

```
cd gazebo-realsense && mkdir build && cd build && cmake .. && make && make install
```

```
git clone https://github.com/SyrianSpock/realsense_gazebo_plugin.git
```

```
git clone https://github.com/bmagyar/icub-moveit.git -b kinetic-devel
```

```
git clone https://github.com/be2rlab/myicub_ros.git
```

**The step inside docker!**

```
source /opt/ros/melodic/setup.bash && cd /user/rosicub && catkin build -j16
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

Run joint control ros-interface (**rosservice call /state_command ...**)

```
rosrun myicub_ros myicub_ros
```

```
rosrun myicub_ros ar_detector_node
```

```
rosrun myicub_ros look_around.py
```





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
