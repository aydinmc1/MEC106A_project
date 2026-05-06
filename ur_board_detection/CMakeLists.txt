cmake_minimum_required(VERSION 3.8)
project(ur_board_detection)

# ament_cmake + ament_cmake_python lets us generate custom interfaces AND
# install Python nodes in a single package without splitting into two.

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# --------------------------------------------------------------------------
# Custom service interfaces
# rosidl_generate_interfaces produces both C++ and Python bindings.
# DetectBoard.srv uses only primitive types so no extra DEPENDENCIES needed.
# --------------------------------------------------------------------------
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/DetectBoard.srv"
)

# --------------------------------------------------------------------------
# Python package installation
# Reads setup.py to:
#   - Install ur_board_detection/ as an importable Python package.
#   - Create console-script executables (from entry_points) under
#     ${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME}/ which is on the ROS2 PATH.
# --------------------------------------------------------------------------
ament_python_install_package(${PROJECT_NAME})

ament_package()
