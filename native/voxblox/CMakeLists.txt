cmake_minimum_required(VERSION 3.20)
project(safety_voxblox LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include(FetchContent)

# Pin upstream sources. SOURCE_SUBDIR bypasses their ROS/catkin build files.
FetchContent_Declare(voxblox_source
  GIT_REPOSITORY https://github.com/ethz-asl/voxblox.git
  GIT_TAG c8066b04075d2fee509de295346b1c0b788c4f38
  SOURCE_SUBDIR unused_ros_build)
FetchContent_Declare(minkindr_source
  GIT_REPOSITORY https://github.com/ethz-asl/minkindr.git
  GIT_TAG 564f12639a8447d4d3e5e7707851424302941056
  SOURCE_SUBDIR unused_ros_build)
FetchContent_MakeAvailable(voxblox_source minkindr_source)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(glog REQUIRED CONFIG)
find_package(Protobuf REQUIRED)
find_package(Threads REQUIRED)
set(PYBIND11_FINDPYTHON ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 2.12 CONFIG QUIET)
if(NOT pybind11_FOUND)
  FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git GIT_TAG v2.13.6)
  FetchContent_MakeAvailable(pybind11)
endif()

set(VB "${voxblox_source_SOURCE_DIR}/voxblox")
# Upstream's parallel-ray guard compares abs(delta) < 0, so axis-aligned
# rays divide 0/0 and never leave their start voxel. Fix only those guards
# in a build-local copy; keep downloaded upstream sources untouched.
file(READ "${VB}/src/integrator/integrator_utils.cc" RAY_SOURCE)
foreach(AXIS x y z)
  string(REPLACE "std::abs(ray_scaled.${AXIS}()) < 0.0"
                 "std::abs(ray_scaled.${AXIS}()) == 0.0" RAY_SOURCE "${RAY_SOURCE}")
endforeach()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/integrator_utils_parallel_fix.cc" "${RAY_SOURCE}")
set(PROTO_ROOT "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${PROTO_ROOT}")
set(PROTO_SRCS)
foreach(NAME Block Layer)
  add_custom_command(
    OUTPUT "${PROTO_ROOT}/voxblox/${NAME}.pb.cc" "${PROTO_ROOT}/voxblox/${NAME}.pb.h"
    COMMAND protobuf::protoc --cpp_out "${PROTO_ROOT}" -I "${VB}/proto" "${VB}/proto/voxblox/${NAME}.proto"
    DEPENDS "${VB}/proto/voxblox/${NAME}.proto" VERBATIM)
  list(APPEND PROTO_SRCS "${PROTO_ROOT}/voxblox/${NAME}.pb.cc")
endforeach()
add_library(safety_voxblox_core STATIC ${PROTO_SRCS}
  "${VB}/src/core/block.cc" "${VB}/src/core/tsdf_map.cc"
  "${VB}/src/integrator/tsdf_integrator.cc" "${CMAKE_CURRENT_BINARY_DIR}/integrator_utils_parallel_fix.cc"
  "${VB}/src/integrator/esdf_integrator.cc" "${VB}/src/utils/neighbor_tools.cc"
  "${VB}/src/utils/timing.cc" "${VB}/src/utils/voxel_utils.cc")
target_include_directories(safety_voxblox_core PUBLIC "${VB}/include" "${PROTO_ROOT}"
  "${minkindr_source_SOURCE_DIR}/minkindr/include")
target_link_libraries(safety_voxblox_core PUBLIC Eigen3::Eigen glog::glog protobuf::libprotobuf Threads::Threads)
pybind11_add_module(_voxblox_native bindings.cpp)
target_link_libraries(_voxblox_native PRIVATE safety_voxblox_core)
set_target_properties(_voxblox_native PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../real_scripts")
