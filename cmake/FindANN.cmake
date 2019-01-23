# Find Ann
#
# This sets the following variables:
# ANN_FOUND - True if FLANN was found.
# ANN_INCLUDE_DIRS - Directories containing the FLANN include files.
# ANN_LIBRARIES - Libraries needed to use FLANN.

find_path(ANN_INCLUDE_DIRS ANN/ANN.h)
find_library(ANN_LIBRARIES ann)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ANN
    DEFAULT_MSG
    ANN_LIBRARIES
    ANN_INCLUDE_DIRS
)
