#!/usr/bin/env python

import rospy

from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState
from std_msgs.msg import Float64MultiArray

class ModelsPositions(object):

    def __init(self):
        self.aarcodes_pub = rospy.Publisher("/arcodes", Float64MultiArray, queue_size=1)
        self.objects_pub = rospy.Publisher("/objects", Float64MultiArray, queue_size=1)

        rospy.wait_for_service('/gazebo/get_model_state')

        self.root_link = [0, 0, 0, 0, 0, 0, 0]
        self.arcode0, = [0, 0, 0, 0, 0, 0, 0]
        self.arcode1 = [0, 0, 0, 0, 0, 0, 0]
        self.arcode2 = [0, 0, 0, 0, 0, 0, 0]
        self.arcode3 = [0, 0, 0, 0, 0, 0, 0]
        self.arcode4 = [0, 0, 0, 0, 0, 0, 0]
        self.arcode5 = [0, 0, 0, 0, 0, 0, 0]
        self.arcode6 = [0, 0, 0, 0, 0, 0, 0]
        self.arcode7 = [0, 0, 0, 0, 0, 0, 0]

        self.banana = [0, 0, 0, 0, 0, 0, 0]
        self.egg = [0, 0, 0, 0, 0, 0, 0]
        self.can = [0, 0, 0, 0, 0, 0, 0]
        self.box = [0, 0, 0, 0, 0, 0, 0]
        self.cube = [0, 0, 0, 0, 0, 0, 0]
        self.rubic = [0, 0, 0, 0, 0, 0, 0]


    def get_position(self, pose):
        return [pose.pose.position.x,
                pose.pose.position.y,
                pose.pose.position.z,
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w]

    def loop(self):
        self.aarcodes_pub = rospy.Publisher("/arcodes", Float64MultiArray, queue_size=1)
        self.objects_pub = rospy.Publisher("/objects", Float64MultiArray, queue_size=1)

        r = rospy.Rate(30)
        while not rospy.is_shutdown():

            # GET POSITIONS OF OBJECT IN A SCENE
            try:
                state = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
                
                self.root_link = self.get_position(state('myicub_fixed', ''))
                self.arcode0 = self.get_position(state('Marker0', ''))
                self.arcode1 = self.get_position(state('Marker1', ''))
                self.arcode2 = self.get_position(state('Marker2', ''))
                self.arcode3 = self.get_position(state('Marker3', ''))
                self.arcode4 = self.get_position(state('Marker4', ''))
                self.arcode5 = self.get_position(state('Marker5', ''))
                self.arcode6 = self.get_position(state('Marker6', ''))
                self.arcode7 = self.get_position(state('Marker7', ''))

                self.banana = self.get_position(state('Banana', ''))
                self.egg = self.get_position(state('Banana_0', ''))
                self.can = self.get_position(state('Can', ''))
                self.box = self.get_position(state('Rubic', ''))
                self.cube = self.get_position(state('Cube', ''))

                arcodes_msg = Float64MultiArray()
                arcodes_msg.layout.data_offset = 7
                arcodes_msg.data = self.arcode0 + self.arcode1 + \
                                    self.arcode2 + self.arcode3 + \
                                    self.arcode4 + self.arcode5 + \
                                    self.arcode6 + self.arcode7

                self.aarcodes_pub.publish(arcodes_msg)
                
                objects_msg = Float64MultiArray()
                objects_msg.layout.data_offset = 7
                objects_msg.data = self.banana + \
                                    self.egg + \
                                    self.can + \
                                    self.box + \
                                    self.cube

                self.objects_pub.publish(objects_msg)

                print()
                print(self.root_link)
                print(self.arcode0)
                print(self.arcode1)
                print(self.arcode2)
                print(self.arcode3)
                print(self.arcode4)
                print(self.arcode5)
                print(self.arcode6)
                print(self.arcode7)
                
            except rospy.ServiceException, e:
                print("Service  call faild: " + e)

            r.sleep()


if __name__ == "__main__":
    rospy.init_node("gazebo_trick_node")
    
    mp = ModelsPositions()
    mp.loop()