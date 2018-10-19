set(
        TEST_SRC
        classifier_test.cc
)

set(Boost_USE_STATIC_LIBS ON)
find_package(Boost REQUIRED COMPONENTS system filesystem)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(classifier_test ${TEST_SRC})
target_link_libraries(classifier_test runerec gtest gmock ${ALL_LIB} ${Boost_LIBRARIES})
TARGET_COMPILE_DEFINITIONS(classifier_test PRIVATE gtest_disable_pthreads=OFF)

