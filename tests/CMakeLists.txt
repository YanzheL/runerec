set(
        TEST_SRC
        classifier_test.cc
        ../Classifiers/Classifier.cc
        ../Classifiers/LocalOpenCVDnnDigitClassifier.cc
        ../Utils.cc
)

if (USE_CUDA)
    set(
            TEST_SRC
            ${TEST_SRC}
            ../Classifiers/LocalTensorRTDigitClassifier.cc
            ../Splitters/LedRuneSplitter.cc
            ../Splitters/FireRuneSplitter.cc
            ../Splitters/RuneSplitter.cc
            ../Splitters/PureRuneSplitter.cc
    )
endif ()

set(Boost_USE_STATIC_LIBS ON)
find_package(Boost REQUIRED COMPONENTS system filesystem)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(classifier_test ${TEST_SRC})
target_link_libraries(classifier_test gtest gmock ${ALL_LIB} ${Boost_LIBRARIES})
TARGET_COMPILE_DEFINITIONS(classifier_test PRIVATE gtest_disable_pthreads=OFF)

