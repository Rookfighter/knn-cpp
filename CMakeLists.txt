# CMakeLists.txt
#
#     Author: Fabian Meyer
# Created On: 08 Nov 2018

cmake_minimum_required(VERSION 3.5)
project(kdtree-eigen)

SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake)

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-Wall -Wextra")
endif(CMAKE_COMPILER_IS_GNUCXX)

# Find Flann
find_package(Flann)
# Find ANN
# find_package(ANN)

# find OpenMP
find_package(OpenMP)
if(OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif(OPENMP_FOUND)

add_subdirectory(dep)

include_directories(
    include
    ${EIGEN3_INCLUDE_DIR}
    ${CATCH2_INCLUDE_DIR}
)

set(LIBS "")

if(${FLANN_FOUND})
    include_directories(${FLANN_INCLUDE_DIRS})
    list(APPEND LIBS ${FLANN_LIBRARIES})
endif(${FLANN_FOUND})
if(${ANN_FOUND})
    include_directories(${ANN_INCLUDE_DIRS})
    list(APPEND LIBS ${ANN_LIBRARIES})
endif(${ANN_FOUND})

install(FILES
    "include/kdtree_ann.h"
    "include/kdtree_eigen.h"
    "include/kdtree_flann.h"
    DESTINATION include)

if(${BUILD_TESTS})
    # Compile tests
    set(TEST_SRC
        "test/main.cpp"
        "test/test_kdtree_eigen.cpp"
    )
    if(${FLANN_FOUND})
        list(APPEND TEST_SRC "test/test_kdtree_flann.cpp")
    endif(${FLANN_FOUND})
    if(${ANN_FOUND})
        list(APPEND TEST_SRC "test/test_kdtree_ann.cpp")
    endif(${ANN_FOUND})

    add_executable(run_tests ${TEST_SRC})
    if(${LIBS})
        target_link_libraries(run_tests ${LIBS})
    endif(${LIBS})
endif(${BUILD_TESTS})
