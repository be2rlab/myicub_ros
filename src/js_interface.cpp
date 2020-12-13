/**
 *  ATTENTION: WORKS ONLY FOR WHOLE BODY iCUB!!!
 * 
 */
#include "js_interface.h"

JSROSInterface::JSROSInterface() {

    state_srv = nh_.advertiseService("state_command", &JSROSInterface::handler, this);

    /**
     *  TORSO CONFIG
     */ 
    Property config_torso;
    config_torso.put("device", "remote_controlboard");
    config_torso.put("remote", "/icubSim/torso");
    config_torso.put("local", "/test/clientt");
    
    driver_torso.open(config_torso);

    if (!driver_torso.isValid()) {
        printf("torso is not available.\n");
        Network::fini();
    }

    driver_torso.view(position_torso);

    /**
     *  LEFT ARM CONFIG
     */ 
    Property config_left_arm;
    config_left_arm.put("device", "remote_controlboard");
    config_left_arm.put("remote", "/icubSim/left_arm");
    config_left_arm.put("local", "/test/clientl");
    
    driver_left_arm.open(config_left_arm);

    if (!driver_left_arm.isValid()) {
        printf("left_arm is not available.\n");
        Network::fini();
    }

    driver_left_arm.view(position_left_arm);

   /**
     *  RIGHT ARM CONFIG
     */ 
    Property config_right_arm;
    config_right_arm.put("device", "remote_controlboard");
    config_right_arm.put("remote", "/icubSim/right_arm");
    config_right_arm.put("local", "/test/clientr");
    
    driver_right_arm.open(config_right_arm);

    if (!driver_right_arm.isValid()) {
        printf("right arm is not available.\n");
        Network::fini();
    }

    driver_right_arm.view(position_right_arm);


   /**
     *  HEAD CONFIG
     */ 
    Property config_head;
    config_head.put("device", "remote_controlboard");
    config_head.put("remote", "/icubSim/head");
    config_head.put("local", "/test/clienth");
    
    driver_head.open(config_head);

    if (!driver_head.isValid()) {
        printf("head is not available.\n");
        Network::fini();
    }

    driver_head.view(position_head);

    position_torso->getAxes(&joints_torso);
    position_left_arm->getAxes(&joints_left);
    position_right_arm->getAxes(&joints_right);
    position_head->getAxes(&joints_head);

    // Default values setup
    double q_torso0[joints_torso] = {0, 0, 0};
    double q_left0[joints_left] = {-45, 80, 20, 80, 0, 0, 0, 55, 0, 0 ,0 ,0 ,0 ,0 ,0, 0};
    double q_right0[joints_right] = {-45, 80, 20, 80, 0, 0, 0, 55, 0, 0 ,0 ,0 ,0 ,0 ,0, 0};
    double q_head0[joints_head] = {-29 ,-30 ,0 ,0 ,0, 0};

    q_torso = new double[joints_torso];
    q_left = new double[joints_left];
    q_right = new double[joints_right];
    q_head = new double[joints_head];
    
    for (int i = 0; i < joints_left; ++i) {
      if (i < joints_torso) {
        q_torso[i] = q_torso0[i];
      }
      q_left[i] = q_left0[i];
      q_right[i] = q_right0[i];
      if (i < joints_head) {
        q_head[i] = q_head0[i];
      }
    }

    // sending command front
    flag = false;

    yInfo() << "JS_ROS_INTERFACE is initialized!";
}

JSROSInterface::~JSROSInterface() {
    delete q_torso;
    delete q_left;
    delete q_right;
    delete q_head;
    
    driver_torso.close();
    driver_left_arm.close();
    driver_right_arm.close();
    driver_head.close();
}

bool JSROSInterface::handler(myicub_ros::StateCommand::Request& req, myicub_ros::StateCommand::Response& res) {
    yInfo() << "Aww! I got a new pose!";
    for (int i = 0; i < joints_left; ++i) {
      if ((req.allow_body_part[0] == 1) && (i < joints_torso)) {
        q_torso[i] = req.q_torso[i];
      }
      if (req.allow_body_part[1] == 1) q_left[i] = req.q_left[i];
      if (req.allow_body_part[2] == 1) q_right[i] = req.q_right[i];
      if ((req.allow_body_part[3] == 1) && (i < joints_head)) {
        q_head[i] = req.q_head[i];
      }
    }
    flag = false; // allow moving
    return true;
  }

void JSROSInterface::spin() {
  ros::Rate R(30);
  while (nh_.ok()) {
    if (flag == false) {
      position_torso->getAxes(&joints_torso);
      position_left_arm->getAxes(&joints_left);
      position_right_arm->getAxes(&joints_right);
      position_head->getAxes(&joints_head);

      for(int i = 0; i < joints_right; ++i) {
        position_torso->setRefSpeed(i, 50);
        position_left_arm->setRefSpeed(i, 50);
        position_right_arm->setRefSpeed(i, 50);
        if (i < 6) {
          position_head->setRefSpeed(i, 10);
        }
      }

      position_torso->positionMove(q_torso); 
      position_left_arm->positionMove(q_left);   
      position_right_arm->positionMove(q_right);   
      position_head->positionMove(q_head);

      bool done0 = false;
      bool done1 = false;
      bool done2 = false;
      bool done3 = false;
      double t0 = Time::now();
      yInfo() << "My body is moving now...";
      while ((!done0 && !done1 && !done2 && !done3) && (Time::now() - t0 < 30.0)) {
            Time::delay(0.1);
            position_torso->checkMotionDone(&done0);
            position_left_arm->checkMotionDone(&done1);
            position_right_arm->checkMotionDone(&done2);
            position_head->checkMotionDone(&done3);
      }
      flag = true;
      yInfo() << "Waiting for a new pose...";
    }
    ros::spinOnce();
    R.sleep();
  }
}


int main(int argc, char *argv[])
{
    ros::init(argc, argv, "js_ros_interface");
    yarp::os::Network yarp;

    if (!yarp.checkNetwork()) {
        yError() << "YARP doesn't seem to be available";
        return EXIT_FAILURE;
    }

    JSROSInterface js_ros_module;
    js_ros_module.spin();

    return 0;
}
