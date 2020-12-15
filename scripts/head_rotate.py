#!/usr/bin/env python
import rospy
import time
from numpy import cos

from myicub_ros.srv import StateCommand

from std_msgs.msg import Float64
from std_msgs.msg import Int8


class HeadRotate(object):

    def __init__(self):
        self.allow_body_part = [0,1,1,1 ]

        self.pose_seq = [
            {   #pose0
                'allow_body_part': [1,1,1,1],
                'torso': [0, 0, 0],
                'q_left': [-45, 30, 0, 100, 0, 0, 0, 55, 0, 0 ,0 ,0 ,0 ,0 ,0, 0],
                'q_right': [-45, 30, 0, 80, 0, 0, 0, 55, 0, 0 ,0 ,0 ,0 ,0 ,0, 0],
                'q_head': [0, 0 ,0 ,0 ,0, 0],
                't': 1.3
            },
            {   #pose1
                'allow_body_part': [1,1,1,1],
                'torso': [0, 0, 0],
                'q_left': [-45, 30, 0, 100, 0, 0, 0, 55, 0, 0 ,0 ,0 ,0 ,0 ,0, 0],
                'q_right': [-45, 30, 0, 80, 0, 0, 0, 55, 0, 0 ,0 ,0 ,0 ,0 ,0, 0],
                'q_head': [-29, 0 , 25, -13, 0, 0],
                't': 1.2
            },
            {   #pose2
                'allow_body_part': [1,1,1,1],
                'torso': [0, 0, 0],
                'q_left': [-45, 60, 0, 100, 0, 0, 0, 55, 0, 0 ,0 ,0 ,0 ,0 ,0, 0],
                'q_right': [-45, 30, 0, 80, 0, 0, 0, 55, 0, 0 ,0 ,0 ,0 ,0 ,0, 0],
                'q_head': [-29, 0 , 30, -40, 0, 0],
                't': 1.2
            },
            {   #pose3
                'allow_body_part': [1,1,1,1],
                'torso': [40, 0, 0],
                'q_left': [-45, 30, 0, 100, 0, 0, 0, 55, 0, 0 ,0 ,0 ,0 ,0 ,0, 0],
                'q_right': [-10, 20, 20, 80, 0, 0, 0, 55, 0, 0 ,0 ,0 ,0 ,0 ,0, 0],
                'q_head': [-29, 0, -10, -10, 0, 0],
                't': 1.6
            },
            {   #pose4
                'allow_body_part': [1,1,1,1],
                'torso': [-30, 0, 0],
                'q_left': [20, 0, 15, 120, 0, 0, 0, 55, 0, 0 ,0 ,0 ,0 ,0 ,0, 0],
                'q_right': [-10, 20, 20, 80, 0, 0, 0, 55, 0, 0 ,0 ,0 ,0 ,0 ,0, 0],
                'q_head': [0, 0 ,10 , 20, 0, 0],
                't': 1.2
            },     
            {   #pose5
                'allow_body_part': [1,1,1,1],
                'torso': [-50, 0, -10],
                'q_left': [20, 0, 15, 120, 0, 0, 0, 55, 0, 0 ,0 ,0 ,0 ,0 ,0, 0],
                'q_right': [-45, 30, 20, 80, 0, 0, 0, 55, 0, 0 ,0 ,0 ,0 ,0 ,0, 0],
                'q_head': [-29, 0, 50, -40, 0, 0],
                't': 2.0
            },     
            {   #pose0
                'allow_body_part': [1,1,1,1],
                'torso': [0, 0, 0],
                'q_left': [-45, 30, 0, 100, 0, 0, 0, 55, 0, 0 ,0 ,0 ,0 ,0 ,0, 0],
                'q_right': [-45, 30, 0, 80, 0, 0, 0, 55, 0, 0 ,0 ,0 ,0 ,0 ,0, 0],
                'q_head': [0, 0 ,0 ,0 ,0, 0],
                't': 1.3
            }
        ]

    def loop(self):
        i = 0
        i0 = i
        flag = True

            
        rospy.wait_for_service('state_command')
        state_command = rospy.ServiceProxy('state_command', StateCommand)


        r = rospy.Rate(1)
        start_time = rospy.get_time()
        k = 0
        while not rospy.is_shutdown():

            resp = state_command(self.allow_body_part, self.pose_seq[i]['torso'], 
                                                        [-45, 70, 0, 100, 0, 0, 0, 55, 0, 0 ,0 ,0 ,0 ,0 ,0, 0],
                                                        [-45, 70, 0, 100, 0, 0, 0, 55, 0, 0 ,0 ,0 ,0 ,0 ,0, 0],
                                                        [-28, 0, 0 * cos(k), -29, 0, 0])
            break               
            k = k + 3.14
            r.sleep()

if __name__=="__main__":
    rospy.init_node('head_rotate_node')

    hr = HeadRotate()
    hr.loop()
