# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2025 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
cmake_minimum_required( VERSION 3.25 )
include( FetchContent )

# Tell the user what's happening.
message( STATUS "Building Acts as part of the TRACCC project" )

# Declare where to get Acts from.
set( TRACCC_ACTS_SOURCE
   "URL;https://github.com/acts-project/acts/archive/refs/tags/v39.0.0.tar.gz;URL_MD5;a50bf40cb5bc6b831d13ee1eb89b0c14"
   CACHE STRING "Source for Acts, when built as part of this project" )
mark_as_advanced( TRACCC_ACTS_SOURCE )
FetchContent_Declare( Acts SYSTEM ${TRACCC_ACTS_SOURCE} )

# Options used in the build of Acts.
set( ACTS_SETUP_EIGEN3 FALSE CACHE BOOL
   "Do not set up Eigen in the Acts code, we do it in this project" )
set( ACTS_BUILD_PLUGIN_JSON TRUE CACHE BOOL
   "Build JSON plugin in Acts" )

# Get it into the current directory.
FetchContent_MakeAvailable( Acts )
