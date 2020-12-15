#!/usr/bin/env python
import rospy
import time
from numpy import array, zeros

from myicub_ros.msg import CameraFeatures

from std_msgs.msg import Float64
from std_msgs.msg import Int8


class CartesianMapping(object):

    def __init__(self):
        # self.pubJointState = rospy.Publisher("joint_states", JointState, queue_size=1)
        self.subJointState = rospy.Subscriber("/marker/features", CameraFeatures, self.callback)

        self.markers_list = zeros((8, 10))  # HERE is number from CameraFeatures float64[80] == 7 x 8*10
        self.markers_looked = zeros(8)


    def parse_marker_pose(self, poses):
        res = []
        N = len(poses)
        poses_list = poses.reshape(8, N / 8)    # because 8 numbers are (id x y z   i j k w)
        for pose in poses_list:
            if sum(pose) != 0: # TODO it is bad but wait...it works
                res.append(pose.tolist())
        return res

    def callback(self, msg):
        camera_psoe = msg.camera_pose
        markers_poses = self.parse_marker_pose(array(msg.marker_pose))
        if markers_poses:
            for p in markers_poses:
                print(sum(self.markers_looked))
                if 0 <= p[0] < 8:
                    self.markers_list[int(p[0])][:] = p
                    self.markers_looked[int(p[0])] = 1
                if all(self.markers_looked):
                    print(self.markers_list)

                    self.make_map()

                    self.markers_looked = zeros(8)  # break like confition

    def make_map(self):
        pass

    def loop(self):
        r = rospy.Rate(30)
        start_time = rospy.get_time()
        while not rospy.is_shutdown():

            r.sleep()


if __name__=="__main__":
    rospy.init_node('cartesian_mapping_node')

    cm = CartesianMapping()
    cm.loop()
