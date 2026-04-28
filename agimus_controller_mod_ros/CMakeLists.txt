cmake_minimum_required(VERSION 3.22.1)

project(agimus_controller_mod_ros LANGUAGES CXX)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

generate_parameter_module(
  trajectory_parameters
  agimus_controller_mod_ros/trajectories/trajectory_parameters.yaml)

ament_python_install_package(${PROJECT_NAME} SCRIPTS_DESTINATION
                             lib/${PROJECT_NAME})

ament_auto_package()
