#ifndef H_POSITION_CONTROL_MATLAB_
#define H_POSITION_CONTROL_MATLAB_

#include <ros/ros.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <string>
#include <cstdio>

#include "moveit_msgs/DisplayTrajectory.h"
#include "moveit_msgs/RobotTrajectory.h"
#include "trajectory_msgs/JointTrajectoryPoint.h"
#include "moveit_msgs/RobotTrajectory.h"
#include "std_msgs/Float64.h"


 #include <yarp/os/Network.h>
 #include <yarp/dev/ControlBoardInterfaces.h>
 #include <yarp/dev/PolyDriver.h>
 #include <yarp/os/Time.h>
 #include <yarp/sig/Vector.h>
 #include <yarp/os/LogStream.h>
 #include <yarp/dev/GazeControl.h>
 #include <yarp/sig/Matrix.h>
 #include <yarp/math/Math.h>

 
 using namespace std;
 using namespace yarp::dev;
 using namespace yarp::sig;
 using namespace yarp::os;
 using namespace yarp::math;

class PositionControlMatlab {
private:

  ros::NodeHandle nh_;
  ros::Subscriber sub;
  ros::Publisher pub;

  std::vector<trajectory_msgs::JointTrajectoryPoint> traj1;
  IPositionControl *pos;
  PolyDriver robotDevice;
  bool motiondone;
    //std::vector<trajectory_msgs::JointTrajectoryPoint> points;

public:
  PositionControlMatlab();
  ~PositionControlMatlab();

  void spin();

    void callback(const trajectory_msgs::JointTrajectoryPoint::ConstPtr& traj1);


};

#endif //H_POSITION_CONTROL_MATLAB_
