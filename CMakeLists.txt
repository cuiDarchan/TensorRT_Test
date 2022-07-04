cmake_minimum_required(VERSION 3.0.0)
project(TensorRT_Test VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3  -pthread -fopenmp")
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV REQUIRED)
# CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
message("CUDA_TOOLKIT_ROOT_DIR = ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "Find CUDA include at ${CUDA_INCLUDE_DIRS}")
message(STATUS "Find CUDA libraries: ${CUDA_LIBRARIES}")

# TensorRT
set(TENSORRT_ROOT /usr/local/TensorRT/TensorRT-7.1.3.4)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} PATH_SUFFIXES include/)
message(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
message(STATUS "Found nvinfer at ${TENSORRT_LIBRARY_INFER}")

find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_ONNXPARSER})
message(STATUS "Find TensorRT libs: ${TENSORRT_LIBRARY}")
include_directories(${TENSORRT_ROOT}/include)
include_directories(/usr/include)
link_directories(${TENSORRT_ROOT}/lib)
link_directories(./)

set(lib_DIR /usr/lib/x84_64-linux-gnu)
link_directories(${lib_DIR})

include_directories(${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${COMMON_INCLUDE} ${JSON_INCLUDE})
link_libraries(nvinfer nvonnxparser ${CUDA_LIBRARIES})

add_executable(build_model build_model.cc) 
target_link_libraries(build_model ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES} ${TENSORRT_LIBRARY} ${Boost_LIBRARIES} ${OpenCV_LIBRARIES} nvinfer nvonnxparser) 

add_executable(model_infer model_infer.cc) 
target_link_libraries(model_infer ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES} ${TENSORRT_LIBRARY} ${Boost_LIBRARIES} ${OpenCV_LIBRARIES} nvinfer nvonnxparser) 