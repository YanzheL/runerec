cmake_minimum_required(VERSION 3.5)
project(runerec)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3 -Werror")
set(USE_REMOTE OFF)
set(USE_CUDA OFF)
set(USE_CAFFE ON)
set(ENABLE_TESTS ON)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
if (USE_EIGEN3)
    find_package(Eigen3 REQUIRED)
    include_directories(${EIGEN3_INCLUDE_DIR})
endif ()


set(
        ADD_SRC
        factory.cc
        factory.h
        Classifiers/LocalOpenCVDnnDigitClassifier.cc
        Utils.cc
)

set(ALL_LIB pthread ${OpenCV_LIBRARIES})

if (USE_CUDA)
    find_package(CUDA REQUIRED)
    add_definitions(-DUSE_CUDA)
    set(
            CUDA_NVCC_FLAGS
            ${CUDA_NVCC_FLAGS};
            -O3
            -gencode arch=compute_61,code=sm_61
            -gencode arch=compute_62,code=sm_62
    )
    set(
            ADD_SRC
            ${ADD_SRC}
            Classifiers/LocalTensorRTDigitClassifier.cc
            Splitters/LedRuneSplitter.cc
            Splitters/FireRuneSplitter.cc
            Splitters/RuneSplitter.cc
            Splitters/PureRuneSplitter.cc
    )
    include_directories(/usr/include/x86_64-linux-gnu)
    include_directories(${CUDA_INCLUDE_DIRS})
    set(ALL_LIB ${ALL_LIB} ${CUDA_LIBRARIES} nvinfer nvinfer_plugin nvparsers)
elseif (USE_CAFFE)
    add_definitions(-DUSE_CAFFE)
elseif (USE_REMOTE)
    add_definitions(-DUSE_REMOTE)
    set(Protobuf_USE_STATIC_LIBS ON)
    set(PROTO_HDRS proto/Image.pb.h proto/RpcMessage.pb.h)
    set(PROTO_SRCS proto/Image.pb.cc proto/RpcMessage.pb.cc)
    find_package(Protobuf REQUIRED)
    include_directories(${Protobuf_INCLUDE_DIRS})
    set(Boost_USE_STATIC_LIBS ON)
    find_package(Boost REQUIRED COMPONENTS system filesystem)
    include_directories(${Boost_INCLUDE_DIRS})
    set(ADD_SRC ${PROTO_SRCS} ${PROTO_HDRS} ${ADD_SRC} Communicator.cpp Communicator.h)
    set(ALL_LIB ${Protobuf_LIBRARIES} ${ALL_LIB})
endif ()

add_library(runerec STATIC ${ADD_SRC})
target_link_libraries(runerec ${ALL_LIB})
add_executable(runerec_main main.cc)
target_link_libraries(runerec_main runerec)

if (ENABLE_TESTS)
    add_subdirectory(googletest)
    add_subdirectory(tests)
endif ()

