#ifndef H_JS_ROS_INTERFACE_
#define H_JS_ROS_INTERFACE_

#include <ros/ros.h>

#include <std_msgs/Float64MultiArray.h>
#include <myicub_ros/StateCommand.h>

#include <vector>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <string>
#include <cstdio>

// #include "moveit_msgs/DisplayTrajectory.h"
// #include "moveit_msgs/RobotTrajectory.h"
// #include "trajectory_msgs/JointTrajectoryPoint.h"
// #include "moveit_msgs/RobotTrajectory.h"
// #include "std_msgs/Float64.h"



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

class JSROSInterface {
  private:
    ros::NodeHandle nh_;
    ros::ServiceServer state_srv;

    PolyDriver driver_torso;
    IPositionControl *position_torso;

    PolyDriver driver_left_arm;
    IPositionControl *position_left_arm;

    PolyDriver driver_right_arm;
    IPositionControl *position_right_arm;

    PolyDriver driver_head;
    IPositionControl *position_head;

    // double q_left[16], q_right[16], q_head[6]; // oouu my god
    double *q_torso, *q_left, *q_right, *q_head;
    int joints_torso, joints_left, joints_right, joints_head;

    bool flag;  // trigger for allow motion

  public:
    JSROSInterface();
    ~JSROSInterface();

    void spin();
    bool handler(myicub_ros::StateCommand::Request& req, myicub_ros::StateCommand::Response& res);
};

#endif //H_JS_ROS_INTERFACE_
