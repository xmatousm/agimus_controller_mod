cmake_minimum_required(VERSION 3.22.1)
project(agimus_controller_mod_msgs)

# Find required packages
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# Define message files
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TrajectoryGoal.msg"
  "msg/RobotHWState.msg"
  "msg/HoleNeeded.msg"
  "action/TrajectoryAction.action"
  DEPENDENCIES geometry_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
