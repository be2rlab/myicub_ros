cmake_minimum_required(VERSION 3.5)
project(myicub_ros)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  message_generation
  visp_bridge
  tf
)

add_message_files(
  FILES
  CameraFeatures.msg
  Marker.msg
  Markers.msg
)
add_service_files(FILES StateCommand.srv)
generate_messages(DEPENDENCIES std_msgs geometry_msgs myicub_ros)

catkin_package(
    INCLUDE_DIRS include
)

find_package(YARP 3.3.0 REQUIRED)
find_package(ICUBcontrib REQUIRED)
find_package(VISP REQUIRED)

# extend the current search path used by cmake to load helpers
list(APPEND CMAKE_MODULE_PATH ${ICUBCONTRIB_MODULE_PATH})

include(ICUBcontribHelpers)
include(ICUBcontribOptions)
include_directories(
    include
    /usr/include/yarp
    ${catkin_INCLUDE_DIRS}
    ${VISP_INCLUDE_DIRS}
)

icubcontrib_set_default_prefix()

add_executable(${PROJECT_NAME} ${CMAKE_SOURCE_DIR}/src/js_interface.cpp)
target_link_libraries(${PROJECT_NAME} ${YARP_LIBRARIES} ${catkin_LIBRARIES})

add_executable(ar_detector_node src/ar_detector_node.cpp)
target_link_libraries(ar_detector_node ${catkin_LIBRARIES} ${VISP_LIBRARIES})
add_dependencies(ar_detector_node ${catkin_EXPORTED_TARGETS})

# add_executable(image_talker)
# target_sources(image_talker PRIVATE ${CMAKE_SOURCE_DIR}/src/image_talker.cpp)
# target_link_libraries(image_talker PRIVATE YARP::YARP_os
#                                      YARP::YARP_init
#                                      YARP::YARP_sig)

#add_executable(image_read)
#target_sources(image_read PRIVATE ${CMAKE_SOURCE_DIR}/src/image_read.cpp)
#target_link_libraries(image_read PRIVATE YARP::YARP_os
#                                     YARP::YARP_init)