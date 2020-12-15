#!/usr/bin/env python
import rospy
import time

from myicub_ros.srv import StateCommand

from std_msgs.msg import Float64
from std_msgs.msg import Int8


class LookAround(object):

    def __init__(self):
        self.allow_body_part = [1,1,1,1]
        # self.torso = [0, 0, 0]
        # self.q_left = [-45, 80, 0, 80, 0, 0, 0, 55, 0, 0 ,0 ,0 ,0 ,0 ,0, 0]
        # self.q_right = [-45, 80, 20, 80, 0, 0, 0, 55, 0, 0 ,0 ,0 ,0 ,0 ,0, 0]
        # self.q_head = [0, 0 ,0 ,0 ,0, 0]

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

        r = rospy.Rate(30)
        start_time = rospy.get_time()
        while not rospy.is_shutdown():
            
            rospy.wait_for_service('state_command')
            state_command = rospy.ServiceProxy('state_command', StateCommand)

            t = rospy.get_time() - start_time
            print(i, t)
            if i == i0 and t > 100:
                start_time = rospy.get_time()
                continue

            if t < self.pose_seq[i]['t']:
                print(i, t)
                if flag:
                    flag = False
                    resp = state_command(self.allow_body_part, self.pose_seq[i]['torso'], 
                                                                self.pose_seq[i]['q_left'],
                                                                self.pose_seq[i]['q_right'],
                                                                self.pose_seq[i]['q_head'])
            else:
                flag = True
                start_time = rospy.get_time()
                i = i + 1
                if i >= len(self.pose_seq):
                    break

            r.sleep()


if __name__=="__main__":
    rospy.init_node('look_around_node')

    la = LookAround()
    la.loop()
