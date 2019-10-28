include_directories(".")

set(TEST_SRC
    "main.cpp"
    "knn/kdtree_eigen.cpp"
    "knn/brute_force.cpp"
)

# if(${FLANN_FOUND})
#     list(APPEND TEST_SRC "test/test_kdtree_flann.cpp")
# endif(${FLANN_FOUND})

add_executable(run_tests ${TEST_SRC})
if(${LIBS})
    target_link_libraries(run_tests ${LIBS})
endif(${LIBS})

if(${BUILD_PERFORMANCE_TEST})
    add_executable(test_performance test_performance.cpp)
endif(${BUILD_PERFORMANCE_TEST})
