cmake_minimum_required(VERSION 3.17)

project(sandbox)

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# GLOBAL SETTINGS :::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# make a bin folder to host all the executables
make_directory(${PROJECT_SOURCE_DIR}/bin)
make_directory(${PROJECT_SOURCE_DIR}/bin/debug)
make_directory(${PROJECT_SOURCE_DIR}/bin/release)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${PROJECT_SOURCE_DIR}/bin/debug")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PROJECT_SOURCE_DIR}/bin/release")

set(COMMON_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/common")
set(COMMON_DATA_DIRS "${CMAKE_CURRENT_LIST_DIR}/data")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# OPTIONS :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# option(USE_OPENGL   "Use OpenGL" OFF)
# option(USE_BOOST    "Use Boost"  OFF)
# option(USE_VTK      "Use VTK"    OFF)
# option(USE_EIGEN    "Use Eigen"  OFF)

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# SUBDIRECTORIES ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
add_subdirectory("src")