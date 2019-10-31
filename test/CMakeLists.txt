
include_directories(
    "."
    ${CATCH2_INCLUDE_DIR}
    ${FLANN_INCLUDE_DIRS}
)

set(TEST_SRC
    "main.cpp"
    "knn/brute_force.cpp"
    "knn/kdtree_flann.cpp"
    "knn/kdtree_minkowski.cpp"
    "knn/multi_index_hashing.cpp"
)

# if(${FLANN_FOUND})
#     list(APPEND TEST_SRC "test/test_kdtree_flann.cpp")
# endif(${FLANN_FOUND})

add_executable(run_tests ${TEST_SRC})
target_link_libraries(run_tests ${FLANN_LIBRARIES})

if(${BUILD_PERFORMANCE_TEST})
    add_executable(test_performance test_performance.cpp)
endif(${BUILD_PERFORMANCE_TEST})
