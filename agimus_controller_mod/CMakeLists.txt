cmake_minimum_required(VERSION 3.22.1)

project(agimus_controller_mod LANGUAGES CXX)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

ament_python_install_package(${PROJECT_NAME} SCRIPTS_DESTINATION
                             lib/${PROJECT_NAME})

ament_auto_package()
