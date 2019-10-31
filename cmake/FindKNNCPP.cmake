# FindKNNCPP.txt
#
#     Author: Fabian Meyer
# Created On: 21 Aug 2019
#
# Defines
#   KNNCPP_INCLUDE_DIR
#   KNNCPP_FOUND

find_path(KNNCPP_INCLUDE_DIR "knn/brute_force.h"
    HINTS
    "${KNNCPP_ROOT}/include"
    "$ENV{KNNCPP_ROOT}/include"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(KNNCPP DEFAULT_MSG KNNCPP_INCLUDE_DIR)
