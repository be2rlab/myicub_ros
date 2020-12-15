#!/usr/bin/env python3
import numpy
import yarp
from detector.class_detector import Detector

yarp.Network.init()


class CameraYarpInterface(object):

    def __init__(self, left_input_port_name, left_output_port_name, right_input_port_name, right_output_port_name):

        # DETECTOR STUFF
        self.dtcr = Detector()
        self.isTrained = False

        self.train_paths = ['dataset_files/Cube.txt',
                            'dataset_files/Banana.txt',
                            'dataset_files/Box.txt'
                            ]
        self.class_names = ['Cube',
                            'Banana',
                            'Box'
                            ]

        # GETTING IMAGE STUFF
        self._cartesian_goal_port = yarp.BufferedPortBottle()
        self._cartesian_goal_port.open("/left_arm_goal_write")

        self.image_w = 640
        self.image_h = 480

        "LEFT EYE"
        # Prepare ports
        self._left_input_port = yarp.Port()
        self._left_input_port_name = left_input_port_name
        self._left_input_port.open(self._left_input_port_name)

        self._left_output_port = yarp.Port()
        self._left_output_port_name = left_output_port_name
        self._left_output_port.open(self._left_output_port_name)

        # Prepare image buffers
        # Input
        self._left_input_buf_image = yarp.ImageRgb()
        self._left_input_buf_image.resize(self.image_w, self.image_h)
        self._left_input_buf_array = numpy.zeros(
            (self.image_h, self.image_w, 3), dtype=numpy.uint8)
        self._left_input_buf_image.setExternal(
            self._left_input_buf_array.data, self._left_input_buf_array.shape[1], self._left_input_buf_array.shape[0])

        # Output
        self._left_output_buf_image = yarp.ImageRgb()
        self._left_output_buf_image.resize(self.image_w, self.image_h)
        self._left_output_buf_array = numpy.zeros(
            (self.image_h, self.image_w, 3), dtype=numpy.uint8)
        self._left_output_buf_image.setExternal(
            self._left_output_buf_array.data, self._left_output_buf_array.shape[1], self._left_output_buf_array.shape[0])

        "RIGHT EYE"
        # Prepare ports
        self._right_input_port = yarp.Port()
        self._right_input_port_name = right_input_port_name
        self._right_input_port.open(self._right_input_port_name)

        self._right_output_port = yarp.Port()
        self._right_output_port_name = right_output_port_name
        self._right_output_port.open(self._right_output_port_name)

        # Prepare image buffers
        # Input
        self._right_input_buf_image = yarp.ImageRgb()
        self._right_input_buf_image.resize(self.image_w, self.image_h)
        self._right_input_buf_array = numpy.zeros(
            (self.image_h, self.image_w, 3), dtype=numpy.uint8)
        self._right_input_buf_image.setExternal(
            self._right_input_buf_array.data, self._right_input_buf_array.shape[1], self._right_input_buf_array.shape[0])

        # Output
        self._right_output_buf_image = yarp.ImageRgb()
        self._right_output_buf_image.resize(self.image_w, self.image_h)
        self._right_output_buf_array = numpy.zeros(
            (self.image_h, self.image_w, 3), dtype=numpy.uint8)
        self._right_output_buf_image.setExternal(
            self._right_output_buf_array.data, self._right_output_buf_array.shape[1], self._right_output_buf_array.shape[0])

    def run(self):

        while True:
            # Read an image from the port
            self._left_input_port.read(self._left_input_buf_image)

            # HERE THE DETECTOR WORKS

            if self.isTrained:
                dtcr.train(self.train_paths, self.class_names)
                dtcr.save_state('detector_state.pckl')

            new_image, det = dtcr.detect(self._left_input_buf_array)
            print("DETECTOR: ", det)

            resp = dtcr.find_object(object_name, image)
            if resp is not "unknown object" and resp is not "not found":
                x1, y1, x2, y2, confidence, class_index = resp
                print("DETECTOR: ", class_index)

            # Send the result to the output port
            # self._left_output_buf_array[:, :] = self._left_input_buf_array
            self._left_output_buf_array[:, :] = new_image
            self._left_output_port.write(self._left_output_buf_image)

            self._right_output_buf_array[:, :] = self._right_input_buf_array
            self._right_output_port.write(self._right_output_buf_image)

            yarp.delay(2)  # TODO think about it

    def cleanup(self):
        self._left_input_port.close()
        self._left_output_port.close()

        self._right_input_port.close()
        self._right_output_port.close()


if __name__ == "__main__":
    """
        input is /icubSim/cam/{left or right}
        output
            - objects in image space
            - /view01
    """

    con = CameraYarpInterface("/lframe:in", "/lframe:out", "/rframe:in", "/rframe:out")

    try:
        assert yarp.Network.connect("/lframe:out", "/lview01")
        assert yarp.Network.connect("/icubSim/cam/left", "/lframe:in")

        assert yarp.Network.connect("/rframe:out", "/rview01")
        assert yarp.Network.connect("/icubSim/cam/right", "/rframe:in")

        # connection between cv node and arm cartesian controller
        # assert yarp.Network.connect("/left_arm_goal_write", "/left_arm_goal_read")

        con.run()
    finally:
        con.cleanup()
