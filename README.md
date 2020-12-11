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


## How to install

