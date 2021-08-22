# CMakeLists.txt
#
#     Author: Fabian Meyer
# Created On: 19 Jul 2019


add_executable(brute_force_search "brute_force_search.cpp")
add_executable(kdtree_minkowski_search "kdtree_minkowski_search.cpp")

if(KNNCPP_FLANN)

    include_directories(
        ${FLANN_INCLUDE_DIRS}
    )

    add_executable(kdtree_flann_search "kdtree_flann_search.cpp")
    target_link_libraries(kdtree_flann_search ${FLANN_LIBRARIES} ${LZ4_LIBRARY})
    set_target_properties(kdtree_flann_search PROPERTIES COMPILE_FLAGS -DKNNCPP_FLANN)
endif(KNNCPP_FLANN)
