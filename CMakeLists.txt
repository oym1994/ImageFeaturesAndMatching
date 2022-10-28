cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

add_definitions(-std=c++14)

set(CXX_FLAGS "-Wall -g")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS} -g")

project(image_feature_matching)

set(OpenCV_DIR /home/hw-robot/Downloads/opencv/opencv-4.1.0/install/lib/cmake/opencv4)
find_package(OpenCV 4 REQUIRED)
#include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS} /usr/local/opencv410/include)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})


option(USE_DEEP_FEATURES "use deep learning based features or not" ON)

if(USE_DEEP_FEATURES)

  message("USE DEEP FEATURES")
  add_definitions("-D USE_DEEP_FEATURES")

  if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
  set(Torch_DIR "~/libtorch/share/cmake/Torch")
  find_package(Torch REQUIRED)
  set(TensorRT_Dir "~/TensorRT-8.2.3.0")
  set(TensorRT_LIB_Dir "${TensorRT_Dir}/targets/x86_64-linux-gnu/lib")  # need to set your own tensorRT directory
  set(TensorRT_INCLUDE_Dir "${TensorRT_Dir}/include/")  # need to set your own tensorRT directory
  include_directories(${CUDA_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS} ${TensorRT_INCLUDE_Dir})

  elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")  # default path for Nvidia NX or Nvidia AGX
  set(Torch_DIR "~/.local/lib/python3.6/site-packages/torch/share/cmake/Torch")
  find_package(Torch REQUIRED)
  set(TensorRT_LIB_Dir "/usr/lib/aarch64-linux-gnu")
  set(TensorRT_INCLUDE_Dir "/usr/include/aarch64-linux-gnu")  # need to set your own tensorRT directory
  include_directories(${CUDA_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS} ${TensorRT_INCLUDE_Dir})
  endif()

  message("Libtorch include: ${TORCH_INCLUDE_DIRS}")

  find_package(CUDA REQUIRED)

# 添加tensorrt的动态库
  link_libraries(${TensorRT_LIB_Dir}/libnvparsers.so)
  link_libraries(${TensorRT_LIB_Dir}/libnvinfer.so)
  link_libraries(${TensorRT_LIB_Dir}/libnvonnxparser.so)
  include_directories(include)

  add_executable (image_matching src/main.cpp src/tensorRT/superpoint_tensorrt.cpp src/tensorRT/tensorrt_generic.cpp src/tensorRT/superglue_tensorrt.cpp src/tensorRT/chunk.cu)
  target_link_libraries (image_matching ${OpenCV_LIBRARIES} ${TORCH_LIBRARIES})

else()

  message("DONT USE DEEP FEATURES")

  add_executable (image_matching src/main.cpp)
  target_link_libraries (image_matching ${OpenCV_LIBRARIES})

endif()
