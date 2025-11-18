cmake_minimum_required(VERSION 3.0.2)
project(controller_pkg)

find_package(catkin REQUIRED COMPONENTS
        rospy
        std_msgs
        geometry_msgs
        sensor_msgs
        cv_bridge
        image_transport
)

catkin_package()

catkin_install_python(PROGRAMS
        scripts/recorder.py
        scripts/lane_follower_il.py
        scripts/lane_follower_il_and_recorder.py
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})