cmake_minimum_required(VERSION 3.8)
project(ur7e_bringup)

find_package(ament_cmake REQUIRED)

# Install all launch, config, urdf, rviz directories
install(DIRECTORY
  launch
  config
  urdf
  rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
