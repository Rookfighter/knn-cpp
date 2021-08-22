
include_directories(
    "."
    ${CATCH2_INCLUDE_DIR}
    ${FLANN_INCLUDE_DIRS}
)

set(TEST_SRC
    "unit/main.cpp"
    "unit/brute_force.cpp"
    "unit/kdtree_minkowski.cpp"
    "unit/multi_index_hashing.cpp"
    "unit/kdtree_flann.cpp"
)

add_executable(run_tests ${TEST_SRC})
if(KNNCPP_FLANN)
    target_link_libraries(run_tests ${FLANN_LIBRARIES} ${LZ4_LIBRARY})
    set_target_properties(run_tests PROPERTIES COMPILE_FLAGS -DKNNCPP_FLANN)
endif(KNNCPP_FLANN)

if(${BUILD_PERFORMANCE_TESTS})
    add_executable(performance_euclidean "performance/performance_euclidean.cpp")
    target_link_libraries(performance_euclidean ${FLANN_LIBRARIES} ${LZ4_LIBRARY})
    set_target_properties(performance_euclidean PROPERTIES COMPILE_FLAGS -DKNNCPP_FLANN)
    add_executable(performance_hamming "performance/performance_hamming.cpp")
endif(${BUILD_PERFORMANCE_TESTS})
