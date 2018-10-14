cmake_minimum_required(VERSION 3.5)
project(handwriting_flamewords_module_2)

set(CMAKE_CXX_STANDARD 11)
set(USE_REMOTE OFF)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
set(USE_CUDA OFF)
set(USE_CAFFE ON)
find_package(OpenCV REQUIRED)
#find_package(Eigen3  REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
set(ADD_SRC main.cc
        Classifiers/Classifier.cc
        Classifiers/Classifier.h
        Utils.cc
        Utils.h
        Splitters/RuneSplitter.cc
        Splitters/RuneSplitter.h
        Splitters/FireRuneSplitter.cc
        Splitters/FireRuneSplitter.h
        Splitters/PureRuneSplitter.cc
        Splitters/PureRuneSplitter.h
        Classifiers/common.h
        Splitters/LedRuneSplitter.cc
        Splitters/LedRuneSplitter.h
        Classifiers/LocalTFDigitClassifier.h
        Classifiers/LocalOpenCVDnnDigitClassifier.cc
        Classifiers/LocalOpenCVDnnDigitClassifier.h
        )

set(ALL_LIB pthread ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})

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
            Classifiers/LocalTensorRTDigitClassifier.h
    )
    include_directories(/usr/include/x86_64-linux-gnu)
    include_directories(${CUDA_INCLUDE_DIRS})
    set(ALL_LIB ${ALL_LIB} ${CUDA_LIBRARIES} nvinfer nvinfer_plugin nvparsers)
elseif (USE_CAFFE)
    add_definitions(-DUSE_CAFFE)
    set(
            ADD_SRC
            ${ADD_SRC}
            Classifiers/LocalCaffeDigitClassifier.h
    )
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
add_executable(handwriting_flamewords_module ${ADD_SRC})
target_link_libraries(handwriting_flamewords_module ${ALL_LIB})

