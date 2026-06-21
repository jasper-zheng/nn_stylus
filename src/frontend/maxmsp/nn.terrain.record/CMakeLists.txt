# Copyright 2018 The Min-DevKit Authors. All rights reserved.
# Use of this source code is governed by the MIT License found in the License.md file.

cmake_minimum_required(VERSION 3.0)

set(C74_MIN_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../min-api)
include(${C74_MIN_API_DIR}/script/min-pretarget.cmake)

if (APPLE)
	set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum OS X deployment version" FORCE)
endif()

#############################################################
# MAX EXTERNAL
#############################################################


include_directories( 
	"${C74_INCLUDES}"
)



set(
SOURCE_FILES
	nn.terrain.record.cpp
    	"../shared/circular_buffer.h"
	"../shared/utils.h"
	"../shared/min_path.h"
	"../shared/min_dictionary.h"
)


add_library( 
	${PROJECT_NAME} 
	MODULE
	${SOURCE_FILES}
    	"../shared/circular_buffer.h"
	"../shared/utils.h"
	"../shared/min_path.h"
	"../shared/min_dictionary.h"
)
if (APPLE) 
set(AUTHOR_DOMAIN "com.jasperzheng")
set(BUNDLE_IDENTIFIER "nn-terrain-record")
endif()

include(${C74_MIN_API_DIR}/script/min-posttarget.cmake)

# Override min-posttarget's C++17 pin so newer toolchains accept C++20 features.
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)


