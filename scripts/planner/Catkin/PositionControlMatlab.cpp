#include "PositionControlMatlab.h"

PositionControlMatlab::PositionControlMatlab() {
  // initialisations

  sub = nh_.subscribe("/joint1",1000, &PositionControlMatlab::callback, this);
  
  

   Property options;
   options.put("device", "remote_controlboard");
   options.put("local", "/test/client");   //local port names
   options.put("remote", "/icubSim/right_arm"); //where we connect to

  robotDevice.open(options);

   if(!robotDevice.isValid()) {
      printf("Device not available.\n");
      Network::fini();
    }
    robotDevice.view(pos);
    
    yInfo() << "[Position Control] init!";

}

PositionControlMatlab::~PositionControlMatlab() {
  // close ports and somethings
  robotDevice.close();
}

void PositionControlMatlab::spin() {

  ros::Rate R(30);

  while (nh_.ok()) {

   // if (!points.empty()) {
    //  for (int i = 0; i < points.size(); ++i) {
          //std::cout << points[i].positions[0] << std::endl;

     // }

   // }

    ros::spinOnce();
    R.sleep();
  }
}


void PositionControlMatlab::callback(const trajectory_msgs::JointTrajectoryPoint::ConstPtr& traj1) {
 int nj = 0;
  pos->getAxes(&nj);
  yInfo () << nj;
 double qinfo[nj]; 
 double q[nj];
 double qq[nj];
 pos->getTargetPositions(qq);
/* yInfo() << qq[0];     // -29.7938 degrees
 yInfo() << qq[1];     // 29.7938 degrees
 yInfo() << qq[2];     // 0 degrees
 yInfo() << qq[3];     // 44.9772 degrees
 yInfo() << qq[4];     // all next are 0 degrees
  yInfo() << qq[5];
   yInfo() << qq[6];
    yInfo() << qq[7];
     yInfo() << qq[8];
      yInfo() << qq[9];
       yInfo() << qq[10];
        yInfo() << qq[11];
         yInfo() << qq[12];
          yInfo() << qq[13];
           yInfo() << qq[14];
            yInfo() << qq[15]; */
 /*double* q = new double[14];
 double* qq = new double[14];
 pos->getTargetPositions(q);
 yInfo() << &q[1];
 yInfo() << &q[2];
 yInfo() << &q[3];
 yInfo() << &q[4];  */
/* pos->getTargetPositions(qq);
 pos->getTargetPosition(1,q[1]);
 pos->getTargetPosition(2,q[2]);
 pos->getTargetPosition(3,q[3]); 
 yInfo() << *q[1];
 yInfo() << *q[2];
 yInfo() << *q[3];
 yInfo() << *q[4];*/  
 //pos->checkMotionDone( &motiondone);
 //q[0] = -60;  //плечо вперёд-назад(вперёд - отрицательный угол)
  //q[1] = 0;   //плечо вправо-влево
 q[2] = 0;  //плечо вокруг оси
//q[3] = 0;  // локоть
q[4] = 0;
q[5]=0;
q[6]=0;
q[7]=0;
q[8]=0;
q[9]=0;
q[10]=0;
q[11]=0;
q[12]=0;
q[13]=0;
q[14]=0;
q[15]=0;
//q[16]=0;

//pos->positionMove(q);

//q[0]=traj1->positions[1];
//q[1]=traj1->positions[2];
//q[3]=traj1->positions[3];
q[0] = -60;
q[1] = 60;
q[3] = 30;
yInfo() << q[0];
yInfo() << q[1];
yInfo() << q[3];
pos->positionMove(q);   
pos->getTargetPositions(qinfo);

 yInfo() << qinfo[0];     // -29.7938 degrees
 yInfo() << qinfo[1];     // 29.7938 degrees
 yInfo() << qinfo[2];     // 0 degrees
 yInfo() << qinfo[3];     // 44.9772 degrees
 yInfo() << qinfo[4];     // all next are 0 degrees
  yInfo() << qinfo[5];
   yInfo() << qinfo[6];
    yInfo() << qinfo[7];
     yInfo() << qinfo[8];
      yInfo() << qinfo[9];
       yInfo() << qinfo[10];
        yInfo() << qinfo[11];
         yInfo() << qinfo[12];
          yInfo() << qinfo[13];
           yInfo() << qinfo[14];
            yInfo() << qinfo[15]; 

//pos->checkMotionDone( &motiondone);   

//Time::delay(0.04);

/*q[1] = q[1] - 65;
pos->positionMove(q);  */



//Time::delay(10);

//pos->positionMove(qq); 


//int n/1=sizeof(traj1);
//int n2=sizeof(traj2);
//int n4=sizeof(trj4);
//int n1=traj1.size();
//int n2=traj2.size();
//int n4=traj4.size();



  //yarp::os::ResourceFinder rf;
 // int np;
  //points.resize(np);
   //int rows =  sizeof (points) / sizeof (points)[0];

    //bool ok;
    //ok = robotDevice.view(pos);

      //if (!ok) {
      //   printf("Problems acquiring interfaces\n");
      //return 0;}
  //points = msg->trajectory[0].joint_trajectory.points;
  //int n = points.size();
  //yInfo() << n;
  /*
  int nj = 0;
  pos->getAxes(&nj);
  yInfo() << nj;

  Vector command;
  command.resize(nj);
  
  double var[nj];
     for (int j = 0; j < n; ++j){
    for (int i = 0; i < 7; ++i){
      //if (i!=3){
      var[i] = points[j].positions[i]*57;
      std::cout << var[i] << ' ';
      pos->positionMove(var);
      //pos->positionMove(1, 60 * sin(t));
      //}
    }
    std::cout << std::endl;
    
    //pos->positionMove(1, 60 * sin(t));
    //for (int t = 0; t < 10; t++){
    //t = t + 0.2;
    //pos->positionMove(1, 60 * sin(t));
    //pos->positionMove(2, 60 * sin(t));
    //pos->positionMove(3, 60 * sin(t));
    Time::delay(0.04);
  }
};


  /*for (int i = 0; i<7; ++i){
    command=points[i].positions;
  
   pos->positionMove(command.data);
   Time::delay(0.5);
   }
}  */
};
int main(int argc, char *argv[])
{
    
  ros::init(argc, argv, "PositionControlMatlab");

  yarp::os::Network yarp;
  if (!yarp.checkNetwork())
  {
    printf("YARP doesn't seem to be available");
    //yError() << "YARP doesn't seem to be available";
    //return EXIT_FAILURE;
  }
  printf("[main] YARP network is working");
  //yInfo() << "[main] YARP network is working";




  PositionControlMatlab mod;
  mod.spin();
  //int status = mod.runModule(rf);

  return 0;
}
